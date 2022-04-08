#!/usr/bin/env python
import argparse
from threading import Condition, Lock

import os
import shutil
import math
import numpy as np
import magnum as mn
import tf
import rospy
import random
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist, Point, Pose, Quaternion, Vector3
from nav_msgs.msg import Odometry
from tf2_msgs.msg import TFMessage
import habitat_sim
from habitat.config.default import get_config
from habitat.core.simulator import Observations
from habitat.sims import make_sim
from ros_x_habitat.msg import PointGoalWithGPSCompass, DepthImage
from ros_x_habitat.srv import EvalEpisode, ResetAgent, GetAgentTime, Roam
from sensor_msgs.msg import Image, CameraInfo, LaserScan, Imu
from std_msgs.msg import Header, Int16
from src.constants.constants import (
    EvalEpisodeSpecialIDs,
    NumericalMetrics,
    PACKAGE_NAME,
    ServiceNames,
)
from move_base_msgs.msg import MoveBaseActionGoal
from src.envs.habitat_eval_rlenv import HabitatEvalRLEnv
from src.evaluators.habitat_sim_evaluator import HabitatSimEvaluator
import time
from src.utils import utils_logging
from src.utils.utils_visualization import generate_video, observations_to_image_for_roam
from src.measures.top_down_map_for_roam import (
    TopDownMapForRoam,
    add_top_down_map_for_roam_to_config,
)
from src.nodes.ik import ik


class HabitatEnvNode:
    r"""
    A class to represent a ROS node with a Habitat simulator inside.
    The node subscribes to agent command topics, and publishes sensor
    readings to sensor topics.
    """

    def __init__(
        self,
        node_name: str,
        config_paths: str = None,
        enable_physics_sim: bool = True,
        use_continuous_agent: bool = True,
        pub_rate: float = 5.0,
        round: int = 0,
    ):
        r"""
        Instantiates a node incapsulating a Habitat sim environment.
        :param node_name: name of the node
        :param config_paths: path to Habitat env config file
        :param enable_physics_sim: if true, turn on dynamic simulation
            with Bullet
        :param use_continuous_agent: if true, the agent would be one
            that produces continuous velocities. Must be false if using
            discrete simulator
        :pub_rate: the rate at which the node publishes sensor readings
        """
        # precondition check
        if use_continuous_agent:
            assert enable_physics_sim

        # initialize node
        self.node_name = node_name
        rospy.init_node(self.node_name)

        rospy.on_shutdown(self.on_exit_generate_video)

        # set up environment config
        self.config = get_config(config_paths)
        # embed top-down map in config
        self.config.defrost()
        self.config.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
        self.config.freeze()
        add_top_down_map_for_roam_to_config(self.config)

        if round == 1:
            random_target = [4, 5, 1]
        elif round == 2:
            random_target = [3, 5, 2]
        elif round == 3:
            random_target = [1, 3, 5]
        else:
            random_target = random.sample(range(1, 6), 3)
        
        self.listExchanges = random_target

        counter = 1
        for i in random_target:
            shutil.copyfile("./data/objects/ycb/configs_convex/letters/"+str(i)+".glb", "./data/objects/ycb/configs_convex/random/"+str(counter)+".glb")
            counter += 1

        # instantiate environment
        self.enable_physics_sim = enable_physics_sim
        self.use_continuous_agent = use_continuous_agent
        self.arm_action_cfg = None
        self.use_ik = True
        self.ee_pos = [90, 0]
        self.switch = 0
        self.target = None
        self.init_move_pos = [0, 0, 0]
        self.move_pos = [0, 0, 0]
        self.curr_pos = [0, 0, 0]
        self.prev_odom = [0, 0, 0]
        self.current_time = rospy.Time.now()
        self.last_time = rospy.Time.now()
        self.counter = 0
        # overwrite env config if physics enabled
        if self.enable_physics_sim:
            HabitatSimEvaluator.overwrite_simulator_config(self.config)
        # define environment
        self.sim = make_sim(id_sim=self.config.SIMULATOR.TYPE, config=self.config.SIMULATOR)
        self.sim.reconfigure(config=self.config.SIMULATOR)
        self.robot_init_pos = [4.20, 0, 3.50]
        self.robot_init_ang = math.pi/2.0
        self.sim.robot.sim_obj.translate(mn.Vector3(self.robot_init_pos[0], 0.045, self.robot_init_pos[2]))
        self.sim.robot.sim_obj.rotate_y(mn.Rad(self.robot_init_ang))
        self.last_angular_velocity=0.0
        self.goal_position=np.matrix([[self.robot_init_pos[0]], [0.0], [self.robot_init_pos[2]], [1.0]])
        self.goal_rotation=mn.Quaternion(mn.Vector3(0.0, 0.0, 0.0), 0.0)
        self.last_position_x = self.robot_init_pos[0]
        self.last_position_y = self.robot_init_pos[2]
        self.last_th = self.sim.robot.base_rot
        self.last_linear_velocity_x=0.0
        self.last_linear_velocity_y=0.0
        x=self.sim.robot.sim_obj.rotation.vector.x
        y=self.sim.robot.sim_obj.rotation.vector.y
        z=self.sim.robot.sim_obj.rotation.vector.z
        w=self.sim.robot.sim_obj.rotation.scalar
        self.pitch_divide2=math.asin((w * y - z * x))/math.pi * 180
        self.map_orientation=float(self.sim.robot.base_rot)
        if self.pitch_divide2<0 and float(self.map_orientation)<np.pi*2/3 :
            self.map_orientation=-self.map_orientation+2*np.pi

        ao_mgr = self.sim.get_articulated_object_manager()
        self.energy_buff = ao_mgr.add_articulated_object_from_urdf(
            "./data/energy_buff/urdf/energy_buff.urdf", fixed_base=False
        )
        self.energy_buff.translate(mn.Vector3(2.45, 0.05, 1.66))
        #self.energy_buff.rotate_x(mn.Rad(-math.pi/2))
        #self.energy_buff.rotate_y(mn.Rad(-math.pi))

        self.recive_box = ao_mgr.add_articulated_object_from_urdf(
            "./data/buff1/urdf/buff1.urdf", fixed_base=False
        )
        self.recive_box.translate(mn.Vector3(2.45, 0.0, 1.75))
        self.recive_box.rotate_x(mn.Rad(-math.pi/2))
        #self.recive_box.rotate_y(mn.Rad(-math.pi))

        self.base_1 = ao_mgr.add_articulated_object_from_urdf(
            "./data/base.urdf", fixed_base=True
        )
        self.base_1.translate(mn.Vector3(2.45, -0.011, 1.75))
        self.base_1.rotate_x(mn.Rad(-math.pi/2))
        #self.recive_box.rotate_y(mn.Rad(-math.pi))

        self.sim.set_rotation(mn.Quaternion(mn.Vector3(0.707, 0, 0), 0.707), 10)
        self.sim.set_rotation(mn.Quaternion(mn.Vector3(0.707, 0, 0), 0.707), 11)
        self.sim.set_rotation(mn.Quaternion(mn.Vector3(0.707, 0, 0), 0.707), 12)

        self.tf_init_trans = [0.412, 0, 1.662]

        self.dtrans = [-0.23, 0, 0.08]

        self.sim.robot.sim_obj.translate(mn.Vector3(self.dtrans))
        self.energy_buff.translate(mn.Vector3(self.dtrans))
        self.recive_box.translate(mn.Vector3(self.dtrans))
        self.base_1.translate(mn.Vector3(self.dtrans))


        self.sim.set_rotation(mn.Quaternion(mn.Vector3(0, 0, 0), 0), 0)
        self.sim.set_rotation(mn.Quaternion(mn.Vector3(0, 0, 0), 0), 1)
        self.sim.set_rotation(mn.Quaternion(mn.Vector3(0, 0, 0), 0), 2)
        self.sim.set_rotation(mn.Quaternion(mn.Vector3(0, 0, 0), 0), 3)
        self.sim.set_rotation(mn.Quaternion(mn.Vector3(0, 0, 0), 0), 4)
        self.sim.set_rotation(mn.Quaternion(mn.Vector3(0, 0, 0), 0), 5)
        self.sim.set_rotation(mn.Quaternion(mn.Vector3(0, 0, 0), 0), 6)
        self.sim.set_rotation(mn.Quaternion(mn.Vector3(0, 0, 0), 0), 7)
        self.sim.set_rotation(mn.Quaternion(mn.Vector3(0, 0, 0), 0), 8)
        self.sim.set_rotation(mn.Quaternion(mn.Vector3(0, 0, 0), 0), 9)

        self.sim.set_translation(self.sim.get_translation(0) + mn.Vector3(self.dtrans), 0)
        self.sim.set_translation(self.sim.get_translation(0) + mn.Vector3(0, 0.05, 0), 1)
        self.sim.set_translation(self.sim.get_translation(2) + mn.Vector3(self.dtrans), 2)
        self.sim.set_translation(self.sim.get_translation(2) + mn.Vector3(0, 0.1, 0), 3)
        self.sim.set_translation(self.sim.get_translation(4) + mn.Vector3(self.dtrans), 4)
        self.sim.set_translation(self.sim.get_translation(4) + mn.Vector3(0, 0.05, 0), 5)
        self.sim.set_translation(self.sim.get_translation(6) + mn.Vector3(self.dtrans), 6)
        self.sim.set_translation(self.sim.get_translation(6) + mn.Vector3(0, 0.05, 0), 7)
        self.sim.set_translation(self.sim.get_translation(8) + mn.Vector3(self.dtrans), 8)
        self.sim.set_translation(self.sim.get_translation(8) + mn.Vector3(0, 0.05, 0), 9)

        self.tf_init_trans = [self.tf_init_trans[0] + self.dtrans[0], self.tf_init_trans[1] + self.dtrans[1], self.tf_init_trans[2] + self.dtrans[2]]

        # shutdown is set to true by eval_episode() to indicate the
        # evaluator wants the node to shutdown
        self.shutdown_lock = Lock()
        with self.shutdown_lock:
            self.shutdown = False

        # enable_eval is set to true by eval_episode() to allow
        # publish_sensor_observations() and step() to run
        # enable_eval is set to false in one of the three conditions:
        # 1) by publish_and_step_for_eval() after an episode is done;
        # 2) by publish_and_step_for_roam() after a roaming session
        #    is done;
        # 3) by main() after all episodes have been evaluated.
        # all_episodes_evaluated is set to True by main() to indicate
        # no more episodes left to evaluate. eval_episodes() then signals
        # back to evaluator, and set it to False again for re-use
        self.all_episodes_evaluated = False
        self.enable_eval = False
        self.enable_eval_cv = Condition()

        # enable_reset is set to true by eval_episode() or roam() to allow
        # reset() to run
        # enable_reset is set to false by reset() after simulator reset
        self.enable_reset_cv = Condition()
        with self.enable_reset_cv:
            self.enable_reset = False
            self.enable_roam = False
            self.episode_id_last = None
            self.scene_id_last = None

        # agent velocities/action and variables to keep things synchronized
        self.command_cv = Condition()
        with self.command_cv:
            if self.use_continuous_agent:
                self.linear_vel = None
                self.angular_vel = None
            else:
                self.action = None
            self.count_steps = None
            self.new_command_published = False

        self.observations = None

        # timing variables and guarding lock
        self.timing_lock = Lock()
        with self.timing_lock:
            self.t_reset_elapsed = None
            self.t_sim_elapsed = None

        # video production variables
        self.make_video = False
        self.observations_per_episode = []
        self.video_frame_counter = 0
        self.video_frame_period = 1  # NOTE: frame rate defined as x steps/frame

        # set up logger
        self.logger = utils_logging.setup_logger(self.node_name)

        # establish evaluation service server
        self.eval_service = rospy.Service(
            f"{PACKAGE_NAME}/{node_name}/{ServiceNames.EVAL_EPISODE}",
            EvalEpisode,
            self.eval_episode,
        )

        # establish roam service server
        self.roam_service = rospy.Service(
            f"{PACKAGE_NAME}/{node_name}/{ServiceNames.ROAM}", Roam, self.roam
        )

        # define the max rate at which we publish sensor readings
        self.pub_rate = float(pub_rate)

        # environment publish and subscribe queue size
        # TODO: make them configurable by constructor argument
        self.sub_queue_size = 10
        self.pub_queue_size = 10

        # publish to sensor topics
        # we create one topic for each of RGB, Depth and GPS+Compass
        # sensor
        if "HEAD_RGB_SENSOR" in self.config.SIMULATOR.AGENT_0.SENSORS:
            self.pub_rgb = rospy.Publisher("/camera/color/image_raw", Image, queue_size=self.pub_queue_size)
            self.pub_third_rgb = rospy.Publisher("third_rgb", Image, queue_size=self.pub_queue_size)
            self.pub_camera_info_rgb = rospy.Publisher(
                    "/camera/color/camera_info", CameraInfo, queue_size=self.pub_queue_size
                )
        if "HEAD_DEPTH_SENSOR" in self.config.SIMULATOR.AGENT_0.SENSORS:
            if self.use_continuous_agent:
                # if we are using a ROS-based agent, we publish depth images
                # in type Image
                self.pub_depth = rospy.Publisher(
                    "/camera/aligned_depth_to_color/image_raw", Image, queue_size=self.pub_queue_size
                )
                # also publish depth camera info
                self.pub_camera_info_depth = rospy.Publisher(
                    "/camera/aligned_depth_to_color/camera_info", CameraInfo, queue_size=self.pub_queue_size
                )
            else:
                # otherwise, we publish in type DepthImage to preserve as much
                # accuracy as possible
                self.pub_depth = rospy.Publisher(
                    "depth", DepthImage, queue_size=self.pub_queue_size
                )
        if "POINTGOAL_WITH_GPS_COMPASS_SENSOR" in self.config.TASK.SENSORS:
            self.pub_pointgoal_with_gps_compass = rospy.Publisher(
                "pointgoal_with_gps_compass",
                PointGoalWithGPSCompass,
                queue_size=self.pub_queue_size
            )
        self.gps_pub = rospy.Publisher(
                "gps",
                PointGoalWithGPSCompass,
                queue_size=self.pub_queue_size
            )
	# before: imu after: /imu/data_raw
        self.imu_pub = rospy.Publisher("/imu/data_raw", Imu, queue_size=self.pub_queue_size)
        self.imu_broadcaster = tf.TransformBroadcaster()
        # before: ray after: /rplidar/scan
        self.ray_pub = rospy.Publisher("/rplidar/scan", LaserScan, queue_size=self.pub_queue_size)
        self.ray_broadcaster = tf.TransformBroadcaster()
        # before: odom after: /ep/odom
        self.odom_pub = rospy.Publisher("/ep/odom", Odometry, queue_size=self.pub_queue_size)
        self.odom_broadcaster = tf.TransformBroadcaster()
        self.gripper_state = rospy.Publisher("gripper_state", Point, queue_size=self.pub_queue_size)

        self.cube_pos_1 = rospy.Publisher("/pose/cube_1", Pose, queue_size=self.pub_queue_size)
        self.cube_pos_2 = rospy.Publisher("/pose/cube_2", Pose, queue_size=self.pub_queue_size)
        self.cube_pos_3 = rospy.Publisher("/pose/cube_3", Pose, queue_size=self.pub_queue_size)
        self.cube_pos_4 = rospy.Publisher("/pose/cube_4", Pose, queue_size=self.pub_queue_size)
        self.cube_pos_5 = rospy.Publisher("/pose/cube_5", Pose, queue_size=self.pub_queue_size)
        self.target_pos_1 = rospy.Publisher("/position/target_1", Point, queue_size=self.pub_queue_size)
        self.target_pos_2 = rospy.Publisher("/position/target_2", Point, queue_size=self.pub_queue_size)
        self.target_pos_3 = rospy.Publisher("/position/target_3", Point, queue_size=self.pub_queue_size)
        self.ep_pos = rospy.Publisher("/pose/ep_world", Pose, queue_size=self.pub_queue_size)

        self.tf_sub = rospy.Subscriber("/tf", TFMessage, self.tf_callback, queue_size=self.sub_queue_size)

        print("\033[0;37;42mTarget Exchange Markers: \033[0m", self.listExchanges)
        from std_msgs.msg import String
        self.exchange_markers = rospy.Publisher("/judgement/exchange_markers", String, queue_size=self.pub_queue_size)

        self.pre_pose_cube = []
        self.pre_buff_w_energy = []
        self.is_buff_still = True

        # [TODO] Flag, start_time in step(1)
        NaN = float('nan')
        self.markers_time_list = [NaN, NaN, NaN]
        self.start_time = 0
        self.markers_time = rospy.Publisher("/judgement/markers_time", String, queue_size=self.pub_queue_size)

        self.i_step = 0

        # subscribe from command topics
        if self.use_continuous_agent:
            self.sub = rospy.Subscriber(
                "cmd_vel", Twist, self.callback, queue_size=self.sub_queue_size
            )

            self.sub1 = rospy.Subscriber(
                "arm_gripper", Point, self.callback1, queue_size=self.sub_queue_size
            )

            self.sub2 = rospy.Subscriber(
                "arm_position", Pose, self.callback2, queue_size=self.sub_queue_size
            )

            self.sub3 = rospy.Subscriber(
                "cmd_position", Twist, self.callback3,  queue_size=self.sub_queue_size
            )
        else:
            self.sub = rospy.Subscriber(
                "action", Int16, self.callback, queue_size=self.sub_queue_size
            )

        # wait until connections with the agent is established
        self.logger.info("env making sure agent is subscribed to sensor topics...")
        while (
            self.pub_rgb.get_num_connections() == 0
            or self.pub_depth.get_num_connections() == 0
            or self.pub_pointgoal_with_gps_compass.get_num_connections() == 0
        ):
            pass
        self.goal_sub=rospy.Subscriber("gps/goal", MoveBaseActionGoal ,self.callback_gps,queue_size=self.sub_queue_size)
        self.logger.info("env initialized")

    def reset(self):
        r"""
        Resets the agent and the simulator. Requires being called only from
        the main thread.
        """
        # reset the simulator
        with self.enable_reset_cv:
            while self.enable_reset is False:
                self.enable_reset_cv.wait()

            # disable reset
            self.enable_reset = False

            # if shutdown is signalled, return immediately
            with self.shutdown_lock:
                if self.shutdown:
                    return

            # locate the last episode specified
            if self.episode_id_last != EvalEpisodeSpecialIDs.REQUEST_NEXT:
                # iterate to the last episode. If not found, the loop exits upon a
                # StopIteration exception
                last_ep_found = False
                while not last_ep_found:
                    try:
                        self.sim.reconfigure(self.config.SIMULATOR)
                    except StopIteration:
                        self.logger.info("Last episode not found!")
                        raise StopIteration
            else:
                # evaluate from the next episode
                pass

            # initialize timing variables
            with self.timing_lock:
                self.t_reset_elapsed = 0.0
                self.t_sim_elapsed = 0.0

            # ------------ log reset time start ------------
            t_reset_start = time.clock()
            # --------------------------------------------

            # initialize observations
            self.observations = self.sim.reset()

            # ------------  log reset time end  ------------
            t_reset_end = time.clock()
            with self.timing_lock:
                self.t_reset_elapsed += t_reset_end - t_reset_start
            # --------------------------------------------

            # initialize step counter
            with self.command_cv:
                self.count_steps = 0

    def _enable_reset(self, request, enable_roam):
        r"""
        Helper method to set self.episode_id_last, self.scene_id_last,
        enable reset and alert threads waiting for reset to be enabled.
        :param request: request dictionary, should contain field
            `episode_id_last` and `scene_id_last`.
        :param enable_roam: if should enable free-roam mode or not.
        """
        with self.enable_reset_cv:
            # unpack evaluator request
            self.episode_id_last = str(request.episode_id_last)
            self.scene_id_last = str(request.scene_id_last)

            # enable (env) reset
            assert self.enable_reset is False
            self.enable_reset = True
            self.enable_roam = enable_roam
            self.enable_reset_cv.notify()

    def _enable_evaluation(self):
        r"""
        Helper method to enable evaluation and alert threads waiting for evalu-
        ation to be enabled.
        """
        with self.enable_eval_cv:
            assert self.enable_eval is False
            self.enable_eval = True
            self.enable_eval_cv.notify()

    def eval_episode(self, request):
        r"""
        ROS service handler which evaluates one episode and returns evaluation
        metrics.
        :param request: evaluation parameters provided by evaluator, including
            last episode ID and last scene ID.
        :return: 1) episode ID and scene ID; 2) metrics including distance-to-
        goal, success and spl.
        """
        # make a response dict
        resp = {
            "episode_id": EvalEpisodeSpecialIDs.RESPONSE_NO_MORE_EPISODES,
            "scene_id": "",
            NumericalMetrics.DISTANCE_TO_GOAL: 0.0,
            NumericalMetrics.SUCCESS: 0.0,
            NumericalMetrics.SPL: 0.0,
            NumericalMetrics.NUM_STEPS: 0,
            NumericalMetrics.SIM_TIME: 0.0,
            NumericalMetrics.RESET_TIME: 0.0,
        }

        if str(request.episode_id_last) == EvalEpisodeSpecialIDs.REQUEST_SHUTDOWN:
            # if shutdown request, enable reset and return immediately
            with self.shutdown_lock:
                self.shutdown = True
            with self.enable_reset_cv:
                self.enable_reset = True
                self.enable_reset_cv.notify()
            return resp
        else:
            # if not shutting down, enable reset and evaluation
            self._enable_reset(request=request, enable_roam=False)

            # enable evaluation
            self._enable_evaluation()

            # wait for evaluation to be over
            with self.enable_eval_cv:
                while self.enable_eval is True:
                    self.enable_eval_cv.wait()


                    # no episode is evaluated. Toggle the flag so the env node
                    # can be reused
                self.all_episodes_evaluated = False
                return resp

    def roam(self, request):
        r"""
        ROS service handler which allows an agent to roam freely within a scene,
        starting from the initial position of the specified episode.
        :param request: episode ID and scene ID.
        :return: acknowledge signal.
        """
        # if not shutting down, enable reset and evaluation
        self._enable_reset(request=request, enable_roam=True)

        # set video production flag
        self.make_video = request.make_video
        self.video_frame_period = request.video_frame_period

        # enable evaluation
        self._enable_evaluation()

        return True

    def cv2_to_depthmsg(self, depth_img: np.ndarray):
        r"""
        Converts a Habitat depth image to a ROS DepthImage message.
        :param depth_img: depth image as a numpy array
        :returns: a ROS Image message if using continuous agent; or
            a ROS DepthImage message if using discrete agent
        """
        if self.use_continuous_agent:
            # depth reading should be denormalized, so we get
            # readings in meters
            assert self.config.SIMULATOR.HEAD_DEPTH_SENSOR.NORMALIZE_DEPTH is False
            depth_img_in_m = np.squeeze(depth_img, axis=2)
            depth_msg = CvBridge().cv2_to_imgmsg(
                depth_img_in_m.astype(np.float32), encoding="passthrough"
            )
        else:
            depth_msg = DepthImage()
            depth_msg.height, depth_msg.width, _ = depth_img.shape
            depth_msg.step = depth_msg.width
            depth_msg.data = np.ravel(depth_img)
        return depth_msg

    def obs_to_msgs(self, observations_hab: Observations):
        r"""
        Converts Habitat observations to ROS messages.

        :param observations_hab: Habitat observations.
        :return: a dictionary containing RGB/depth/Pos+Orientation readings
        in ROS Image/Pose format.
        """
        observations_ros = {}

        # take the current sim time to later use as timestamp
        # for all simulator readings
        t_curr = rospy.Time.now()

        for sensor_uuid, _ in observations_hab.items():
            sensor_data = observations_hab[sensor_uuid]
            # we publish to each of RGB, Depth and GPS+Compass sensor
            if sensor_uuid == "robot_head_rgb":
                sensor_msg = CvBridge().cv2_to_imgmsg(
                    sensor_data.astype(np.uint8), encoding="rgb8"
                )
            elif sensor_uuid == "robot_arm_rgb":
                sensor_msg = CvBridge().cv2_to_imgmsg(
                    sensor_data.astype(np.uint8), encoding="rgb8"
                )
            elif sensor_uuid == "robot_head_depth":
                sensor_msg = self.cv2_to_depthmsg(sensor_data)
            elif sensor_uuid == "pointgoal_with_gps_compass":
                sensor_msg = PointGoalWithGPSCompass()
                sensor_msg.distance_to_goal = sensor_data[0]
                sensor_msg.angle_to_goal = sensor_data[1]
            # add header to message, and add the message to observations_ros
            if sensor_uuid in ["robot_head_rgb", "robot_arm_rgb", "robot_head_depth", "pointgoal_with_gps_compass"]:
                h = Header()
                h.stamp = t_curr
                h.frame_id = 'camera_color_optical_frame'
                sensor_msg.header = h
                observations_ros[sensor_uuid] = sensor_msg

        return observations_ros

    def publish_sensor_observations(self):
        r"""
        Waits until evaluation is enabled, then publishes current simulator
        sensor readings. Requires to be called 1) after simulator reset and
        2) when evaluation has been enabled.
        """
        # pack observations in ROS message
        observations_ros = self.obs_to_msgs(self.observations)
        for sensor_uuid, _ in self.observations.items():
            # we publish to each of RGB, Depth and Ptgoal/GPS+Compass sensor
            if sensor_uuid == "robot_head_rgb":
                self.pub_rgb.publish(observations_ros["robot_head_rgb"])
            elif sensor_uuid == "robot_arm_rgb":
                self.pub_third_rgb.publish(observations_ros["robot_arm_rgb"])
            elif sensor_uuid == "robot_head_depth":
                self.pub_depth.publish(observations_ros["robot_head_depth"])
                if self.use_continuous_agent:
                    self.pub_camera_info_rgb.publish(
                        self.make_depth_camera_info_msg(
                            observations_ros["robot_head_depth"].header,
                            observations_ros["robot_head_depth"].height,
                            observations_ros["robot_head_depth"].width,
                        )
                    )
                    self.pub_camera_info_depth.publish(
                        self.make_depth_camera_info_msg(
                            observations_ros["robot_head_depth"].header,
                            observations_ros["robot_head_depth"].height,
                            observations_ros["robot_head_depth"].width,
                        )
                    )
            elif sensor_uuid == "pointgoal_with_gps_compass":
                self.pub_pointgoal_with_gps_compass.publish(
                    observations_ros["pointgoal_with_gps_compass"]
                )

    def make_depth_camera_info_msg(self, header, height, width):
        r"""
        Create camera info message for depth camera.
        :param header: header to create the message
        :param height: height of depth image
        :param width: width of depth image
        :returns: camera info message of type CameraInfo.
        """
        # code modifed upon work by Bruce Cui
        camera_info_msg = CameraInfo()
        camera_info_msg.header = header
        fx, fy = width / 2 / math.tan(1.2037/2), width / 2 / math.tan(1.2037/2)
        cx, cy = width / 2, height / 2

        camera_info_msg.width = width
        camera_info_msg.height = height
        camera_info_msg.distortion_model = "plumb_bob"
        camera_info_msg.K = np.float32([fx, 0, cx, 0, fy, cy, 0, 0, 1])
        camera_info_msg.D = np.float32([0, 0, 0, 0, 0])
        camera_info_msg.P = [fx, 0, cx, 0, 0, fy, cy, 0, 0, 0, 1, 0]
        return camera_info_msg
    def rotation_x(self, roll):
        rotation=np.matrix(
        [[1.0,0.0,0.0,0.0],
        [0.0,math.cos(roll),math.sin(roll),0.0],
        [0.0,-math.sin(roll),math.cos(roll),0.0],
        [0.0,0.0,0.0,1.0]])
        return rotation
    def rotation_z(self, yaw):
        rotation=np.matrix(
        [[math.cos(yaw),math.sin(yaw),0.0,0.0],
        [-math.sin(yaw),math.cos(yaw),0.0,0.0],
        [0.0,0.0,1.0,0.0],
        [0.0,0.0,0.0,1.0]])
        return rotation
    def shift_mat(self, x,y,z):
        shift=np.matrix(
        [[1.0,0.0,0.0,x],
        [0.0,1.0,0.0,y],
        [0.0,0.0,1.0,z],
        [0.0,0.0,0.0,1.0]])
        return shift

    def step(self):
        r"""
        Enact a new command and update sensor observations.
        Requires 1) being called only when evaluation has been enabled and
        2) being called only from the main thread.
        """
        if not self.use_continuous_agent:
            # if using Habitat agent, wait for new action before stepping
            with self.command_cv:
                while self.new_command_published is False:
                    self.command_cv.wait()
                self.new_command_published = False

        # enact the action / velocities
        # ------------ log sim time start ------------
        t_sim_start = time.clock()
        # --------------------------------------------

        if self.use_continuous_agent:
            with self.command_cv:
                #if self.switch == 1 and not self.sim.grasp_mgr.is_grasped:
                #    rotation, translation = qr_code_locate(self.observations)
                #    cube_pos = [10, 10, 10]
                #    if len(translation) > 0:
                #        for t in translation:
                #            if t[0][0][2] < cube_pos[0]:
                #                cube_pos[0] = t[0][0][2]
                #                cube_pos[1] = t[0][0][1]
                #                cube_pos[2] = t[0][0][0]
                #        self.target = cube_pos
                #        self.target[0] *= 100
                #        self.target[1] *= -100
                #        self.target[2] *= 100
                #        self.target[0] += 185
                #        self.target[1] -= 50
                #    else:
                #        self.target = None

                #self.env.set_agent_velocities(self.linear_vel, self.angular_vel)

                #time_start = rospy.Time.now()

                x=self.sim.robot.sim_obj.rotation.vector.x
                y=self.sim.robot.sim_obj.rotation.vector.y
                z=self.sim.robot.sim_obj.rotation.vector.z
                w=self.sim.robot.sim_obj.rotation.scalar

                r = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
                r = r / math.pi * 180
                self.pitch_divide2 = math.asin((w * y - z * x))
                self.pitch_divide2 = self.pitch_divide2 / math.pi * 180
                y = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
                y = y / math.pi * 180

#                self.sim.robot.sim_obj.root_linear_velocity
                time_change=rospy.Time.now().to_sec()-self.last_time.to_sec()

                #angular in world 0-pi*4/3, pi*2/3-0
                th_world = float(self.sim.robot.base_rot)
                if self.pitch_divide2<0 and float(th_world)<np.pi*2/3 :
                    th_world=-th_world+2*np.pi#0-2pi

                #angular in map(star from init orientation) -pi/2-pi*3/2
                th_map = float(th_world)-self.map_orientation
                if th_map<0 :
                    th_map=th_map+np.pi*2#0-2pi

                vel_control = habitat_sim.physics.VelocityControl()
                vel_control.controlling_lin_vel = True
                vel_control.controlling_ang_vel = True
                vel_control.lin_vel_is_local = True
                vel_control.ang_vel_is_local = True
                lv = [0, 0, 0]
                av = [0, 0, 0]
                limits=2.0
                if self.linear_vel is not None and self.angular_vel is not None:
                    lv=np.clip(self.linear_vel, -0.5, 0.5)
                    av=np.clip(self.angular_vel, -0.5, 0.5)
                    if lv[0] > -0.1 and lv[0] < 0.1:
                        lv[0] = 0
                    if lv[2] > -0.1 and lv[2] < 0.1:
                        lv[2] = 0
                    if av[1] > -0.01 and av[1] < 0.01:
                        av = mn.Vector3(0, 0, 0)
                    vel_control.linear_velocity = mn.Vector3(-lv[2], 0, lv[0])
                    vel_control.angular_velocity = av

                ctrl_freq = self.pub_rate

                trans = self.sim.robot.sim_obj.transformation
                rigid_state = habitat_sim.RigidState(
                    mn.Quaternion.from_matrix(trans.rotation()), trans.translation
                )

                target_rigid_state = vel_control.integrate_transform(
                    1 / ctrl_freq, rigid_state
                )
                end_pos = self.sim.step_filter(
                    rigid_state.translation, target_rigid_state.translation
                )

                target_trans = mn.Matrix4.from_(
                    target_rigid_state.rotation.to_matrix(), end_pos
                )
                self.sim.robot.sim_obj.transformation = target_trans

                #time_end = rospy.Time.now()
                #print("cmd_vel: ", (time_end - time_start).to_sec())

                #time_start = rospy.Time.now()

                if not self.use_ik:
                    if self.arm_action_cfg is None:
                        self.arm_action_cfg = [0, 0]
                    joint_1_action = np.zeros(12)
                    joint_2_action = np.zeros(12)

                    joint_1_action[0] =  1.0
                    joint_1_action[1] = -1.0
                    joint_1_action[2] = -1.0
                    joint_1_action[3] = -1.0
                    joint_1_action[4] = -1.0

                    joint_2_action[5] = -1.0
                    joint_2_action[6] =  1.0
                    joint_2_action[3] =  1.0
                    joint_2_action[7] = -1.0
                    joint_2_action[4] =  1.0

                    joint_1_action *= self.arm_action_cfg[0]
                    joint_2_action *= self.arm_action_cfg[1]

                    arm_action = joint_1_action + joint_2_action
                    arm_action *= 0.0125

                    self.sim.robot.arm_motor_pos = (
                        arm_action + self.sim.robot.arm_motor_pos
                    )
                else:
                    if self.target is None and self.arm_action_cfg is not None:
                        self.ee_pos[0] = self.arm_action_cfg[0]
                        self.ee_pos[1] = self.arm_action_cfg[1]
                    elif self.target is not None:
                        self.ee_pos[0] = self.target[0]
                        self.ee_pos[1] = self.target[1]
                        self.target = None
                    joint_1_action, joint_2_action, has_solution = ik(self.ee_pos[0], self.ee_pos[1])

                    joint_1_action -= 1.35
                    joint_2_action += 2.15

                    #print(joint_1_action, joint_2_action)

                    joint_1_action = np.clip(joint_1_action - self.sim.robot._get_motor_pos(2), -0.05, 0.05)
                    joint_2_action = np.clip(joint_2_action - self.sim.robot._get_motor_pos(5), -0.05, 0.05)

                    if has_solution:
                        arm_action = np.zeros(12)
                        arm_action[0] = joint_1_action
                        arm_action[1] = -joint_1_action
                        arm_action[2] = -joint_1_action
                        arm_action[3] = joint_2_action
                        arm_action[4] = joint_2_action
                        arm_action[5] = -(joint_2_action + joint_1_action)
                        arm_action[6] = joint_2_action + joint_1_action
                        arm_action[7] = -(joint_2_action + joint_1_action)

                        self.sim.robot.arm_motor_pos = (
                            arm_action + self.sim.robot.arm_motor_pos
                        )

                #time_end = rospy.Time.now()
                #print("arm_pos: ", (time_end - time_start).to_sec())

                #time_start = rospy.Time.now()

                robot_transformation = self.sim.robot.sim_obj.transformation
                ee_link_pos = [self.ee_pos[0]/1000, self.ee_pos[1]/1000 + 0.15-0.03, 0]
                ee_offset = [0.11, -0.04, 0]
                ee_pos = [ee_link_pos[0] + ee_offset[0], ee_link_pos[1] + ee_offset[1], ee_link_pos[2] + ee_offset[2]]

                ee_pos_mat = np.mat([[ee_pos[0]], [ee_pos[1]], [ee_pos[2]], [1]])
                ee_pos_mat_trans = robot_transformation * ee_pos_mat

                unmove_obj = [10, 11, 12, 13, 14, 15]

                if self.switch == 1 and not self.sim.grasp_mgr.is_grasped:
                    scene_obj_pos = self.sim.get_scene_pos()
                    ee_pos = mn.Vector3(ee_pos_mat_trans[0, 0], ee_pos_mat_trans[1, 0], ee_pos_mat_trans[2, 0])
                    if len(scene_obj_pos) != 0:
                        # Get the target the EE is closest to.
                        closest_obj_idx = np.argmin(
                            np.linalg.norm(scene_obj_pos - ee_pos, ord=2, axis=-1)
                        )

                        closest_obj_pos = scene_obj_pos[closest_obj_idx]
                        to_target = np.linalg.norm(ee_pos - closest_obj_pos, ord=2)
                        sim_idx = self.sim.scene_obj_ids[closest_obj_idx]
                        if to_target < 0.04 and sim_idx not in unmove_obj:
                            self.sim.grasp_mgr.snap_to_obj(sim_idx)
                            grip_state = Point()
                            grip_state.x = 1
                            self.gripper_state.publish(grip_state)
                    self.switch = -1
                elif self.switch == 0 and self.sim.grasp_mgr.is_grasped:
                    self.sim.grasp_mgr.desnap()

                #time_end = rospy.Time.now()
                #print("grasp: ", (time_end - time_start).to_sec())

                #time_start = rospy.Time.now()

                if self.counter % self.pub_rate == 0:
                    grip_state = Point()
                    if self.sim.grasp_mgr.is_grasped:
                        grip_state.x = 1
                    else:
                        grip_state.x = 0
                    self.gripper_state.publish(grip_state)
                self.counter += 1

                if self.init_move_pos[0] < 0.1 and self.init_move_pos[0] > -0.1:
                    self.init_move_pos[0] = 0
                if self.init_move_pos[1] < 0.1 and self.init_move_pos[1] > -0.1:
                    self.init_move_pos[1] = 0
                if self.init_move_pos[2] < 0.1 and self.init_move_pos[2] > -0.1:
                    self.init_move_pos[2] = 0
                if self.init_move_pos != [0, 0, 0]:
                    self.move_pos = self.init_move_pos
                    self.init_move_pos = [0, 0, 0]

                move_step = [0, 0, 0]
                trans_mat = np.mat([[math.cos(th_world), 0, math.sin(th_world)], [0, 1, 0], [-math.sin(th_world), 0, math.cos(th_world)]])
                if self.move_pos == [0, 0, 0]:
                    pass
                elif not self.curr_pos == self.move_pos:
                    move_step[0] = np.clip(self.move_pos[0] - self.curr_pos[0], -0.005, 0.005)
                    move_step[1] = np.clip(self.move_pos[1] - self.curr_pos[1], -0.005, 0.005)
                    move_step[2] = np.clip(self.move_pos[2] - self.curr_pos[2], -0.01, 0.01)
                    move_step_mat = np.mat([[move_step[0]], [0], [move_step[1]]])
                    trans_move_step = trans_mat*move_step_mat
                    self.sim.robot.sim_obj.translate(mn.Vector3(trans_move_step[0, 0], trans_move_step[1, 0], trans_move_step[2, 0]))
                    self.sim.robot.sim_obj.rotate_y(mn.Rad(move_step[2]))
                    self.curr_pos = [self.curr_pos[0] + move_step[0], self.curr_pos[1] + move_step[1], self.curr_pos[2] + move_step[2]]
                else:
                    self.move_pos = [0, 0, 0]
                    self.curr_pos = [0, 0, 0]

                #self.sim.robot.sim_obj.translate(
                #        mn.Vector3(
                #            (self.move_pos[0] * math.cos(rotation) + self.move_pos[1] * math.sin(rotation)),
                #            0,
                #            (-self.move_pos[0] * math.sin(rotation) + self.move_pos[1] * math.cos(rotation))
                #        )
                #    )
                #self.move_pos = [0, 0]

                #time_end = rospy.Time.now()
                #print("cmd_pos: ", (time_end - time_start).to_sec())

                #time_start = rospy.Time.now()

                if self.counter % 3 == 0:
                    num_readings=760
                    scan = LaserScan()
                    scan.header.stamp = self.current_time
                    scan.header.frame_id = "laser_link"
                    scan.angle_min = -np.pi
                    scan.angle_max = np.pi
                    scan.angle_increment = 2*np.pi / num_readings
                    scan.time_increment = 1*3/self.pub_rate/num_readings
                    scan.range_min = 0.15
                    scan.range_max = 12.0
                    ray = habitat_sim.geo.Ray()
                    origin = np.mat([[0.12], [0.15], [0]])

                    origin = trans_mat*origin
                    ray.origin = mn.Vector3(origin[0, 0]+self.sim.robot.sim_obj.transformation[3][0], origin[1, 0], origin[2, 0]+self.sim.robot.sim_obj.transformation[3][2])

                    for i in range(num_readings+1,0,-1):
                        ray.direction = mn.Vector3(math.cos(i/num_readings*2*np.pi-th_world),0,math.sin(i/num_readings*np.pi*2-th_world))
                        #print(ray.direction)
                        raycast_results = self.sim.cast_ray(ray, 100)
    #!                	scan.intensities.append(10*i)
                        if len(raycast_results.hits) != 0:
    #                		print(ray.origin)
                            scan.ranges.append(raycast_results.hits[0].ray_distance)
                    #give up 90 angle behind
                    for i in range(int(num_readings*0.375),int(num_readings*0.625), 1):
                        j=i

                        if i>(num_readings-1) :
                            j=i-num_readings
                        scan.ranges[j]=float("inf")
                    self.ray_pub.publish(scan)

                #time_end = rospy.Time.now()
                #print("laser_scan: ", (time_end - time_start).to_sec())

                #time_start = rospy.Time.now()

                imu_data=Imu()
                imu_data.header.stamp= self.current_time
                imu_data.header.frame_id = "imu_link"


                imu_data.orientation_covariance[0] = 0
                imu_data.orientation_covariance[1] = 0
                imu_data.orientation_covariance[2] = 0
                imu_data.orientation_covariance[3] = 0
                imu_data.orientation_covariance[4] = 0
                imu_data.orientation_covariance[5] = 0
                imu_data.orientation_covariance[6] = 0
                imu_data.orientation_covariance[7] = 0
                imu_data.orientation_covariance[8] = 0

                imu_data.angular_velocity_covariance[0]=0
                imu_data.angular_velocity_covariance[1]=0
                imu_data.angular_velocity_covariance[2]=0
                imu_data.angular_velocity_covariance[3]=0
                imu_data.angular_velocity_covariance[4]=0
                imu_data.angular_velocity_covariance[5]=0
                imu_data.angular_velocity_covariance[6]=0
                imu_data.angular_velocity_covariance[7]=0
                imu_data.angular_velocity_covariance[8]=0

                imu_data.linear_acceleration_covariance[0]=0
                imu_data.linear_acceleration_covariance[1]=0
                imu_data.linear_acceleration_covariance[2]=0
                imu_data.linear_acceleration_covariance[3]=0
                imu_data.linear_acceleration_covariance[4]=0
                imu_data.linear_acceleration_covariance[5]=0
                imu_data.linear_acceleration_covariance[6]=0
                imu_data.linear_acceleration_covariance[7]=0
                imu_data.linear_acceleration_covariance[8]=0

                current_position_x=self.sim.robot.sim_obj.transformation[3][0]
                current_position_y=self.sim.robot.sim_obj.transformation[3][2]

                angular_change=th_world-float(self.last_th)

                imu_quat = tf.transformations.quaternion_from_euler(0, 0, th_map)
                imu_data.orientation=Quaternion(*imu_quat)

                x_position_change=self.sim.robot.sim_obj.transformation[3][0]-self.last_position_x
                y_position_change=self.sim.robot.sim_obj.transformation[3][2]-self.last_position_y

                #linear_velocity_x_world=x_position_change*self.pub_rate
                #linear_velocity_y_world=y_position_change*self.pub_rate
                #linear_velocity_x=(linear_velocity_x_world*math.cos(th_world)+linear_velocity_y_world*math.sin(th_world))
                #linear_velocity_y=(linear_velocity_y_world*math.cos(th_world)-linear_velocity_x_world*math.sin(th_world))
                transformation_mat = np.mat([[math.cos(th_world), 0, -math.sin(th_world)], [-math.sin(th_world), 0, -math.cos(th_world)], [0, 1, 0]])

                world_linear_vel_mat = np.mat([[x_position_change*self.pub_rate], [0], [y_position_change*self.pub_rate]])
                world_linear_vel_trans = transformation_mat * world_linear_vel_mat
                linear_velocity_x=world_linear_vel_trans[0, 0]
                linear_velocity_y=world_linear_vel_trans[1, 0]
                imu_data.linear_acceleration.x = (linear_velocity_x-self.last_linear_velocity_x)*self.pub_rate
                imu_data.linear_acceleration.y = (linear_velocity_y-self.last_linear_velocity_y)*self.pub_rate
                imu_data.linear_acceleration.z = 9.8  #TODO: check real imu linear acceleration z

                imu_data.angular_velocity.x = 0
                imu_data.angular_velocity.y = 0
                if angular_change >  6.0 :
                    angular_change=angular_change+np.pi*2
                if angular_change < -6.0 :
                    angular_change=angular_change-np.pi*2


                imu_data.angular_velocity.z = angular_change*self.pub_rate
                if float(imu_data.angular_velocity.z)>limits or float(imu_data.angular_velocity.z)<-limits:
                    imu_data.angular_velocity.z=self.last_angular_velocity
                self.imu_pub.publish(imu_data)

                #time_end = rospy.Time.now()
                #print("imu: ", (time_end - time_start).to_sec())

                #time_start = rospy.Time.now()

                self.last_th = th_world
                self.last_angular_velocity=imu_data.angular_velocity.z
                self.last_position_x = self.sim.robot.sim_obj.transformation[3][0]
                self.last_position_y = self.sim.robot.sim_obj.transformation[3][2]
                self.last_linear_velocity_x=linear_velocity_x
                self.last_linear_velocity_y=linear_velocity_y

                gps_msg = PointGoalWithGPSCompass()
                gps_msg.header.stamp= self.current_time
                gps_msg.header.frame_id = "gps"
                angle_env2base=th_world+self.map_orientation
                rotation_x = self.rotation_x(-np.pi/2)
                rotation_z = self.rotation_z(th_map+self.map_orientation)
                shift_goal = self.shift_mat(-self.sim.robot.sim_obj.transformation[3][0],0.0,-self.sim.robot.sim_obj.transformation[3][2])
                transf = rotation_z*rotation_x*shift_goal
                goal_on_map= transf*self.goal_position
                r"""
                goal_env2base_x=(self.goal_position[2,0]*math.cos(angle_env2base)+self.goal_position[0,0]*math.sin(angle_env2base))-(self.sim.robot.sim_obj.transformation[3][2]*math.cos(angle_env2base)+self.sim.robot.sim_obj.transformation[3][0]*math.sin(angle_env2base))
                goal_env2base_y=(self.goal_position[0,0]*math.cos(angle_env2base)-self.goal_position[2,0]*math.sin(angle_env2base))-(self.sim.robot.sim_obj.transformation[3][0]*math.cos(angle_env2base)-self.sim.robot.sim_obj.transformation[3][2] * math.sin(angle_env2base))
                """
                x_distance=goal_on_map[0,0]
                y_distance=goal_on_map[1,0]
                gps_msg.distance_to_goal = (x_distance**2+y_distance**2)**0.5

                if x_distance==0 :
                    if y_distance > 0 :
                        gps_msg.angle_to_goal=0.0
                    else :
                        gps_msg.angle_to_goal=np.pi
                elif y_distance==0 :
                    if x_distance>0 :
                        gps_msg.angle_to_goal=np.pi/2
                    else :
                        gps_msg.angle_to_goal=np.pi*1.5
                elif x_distance>0 and y_distance>0 :
                    gps_msg.angle_to_goal=float(math.atan(abs(y_distance/x_distance)))
                elif x_distance<0 and y_distance>0 :
                    gps_msg.angle_to_goal=float(np.pi-math.atan(abs(y_distance/x_distance)))
                elif x_distance<0 and y_distance<0 :
                    gps_msg.angle_to_goal=float(np.pi+math.atan(abs(y_distance/x_distance)))
                elif x_distance>0 and y_distance<0 :
                    gps_msg.angle_to_goal=float(np.pi*2-math.atan(abs(y_distance/x_distance)))

#                print("d:",gps_msg.distance_to_goal)
#                print("a:",gps_msg.angle_to_goal)
                self.gps_pub.publish(gps_msg)

                #time_end = rospy.Time.now()
                #print("gps: ", (time_end - time_start).to_sec())

                #time_start = rospy.Time.now()

                self.current_time = rospy.Time.now()

                movement_x = self.sim.robot.sim_obj.transformation[3][0] - self.robot_init_pos[0]
                movement_z = self.sim.robot.sim_obj.transformation[3][2] - self.robot_init_pos[2]
                odom_th = th_map

                transformation_mat = np.mat([[math.cos(self.robot_init_ang), 0, -math.sin(self.robot_init_ang)], [-math.sin(self.robot_init_ang), 0, -math.cos(self.robot_init_ang)], [0, 1, 0]])

                curr_pos_trans = transformation_mat * np.mat([[movement_x], [0], [movement_z]])

                odom_x = curr_pos_trans[0, 0]
                odom_y = curr_pos_trans[1, 0]

                odom_quat = tf.transformations.quaternion_from_euler(0, 0, odom_th)

                vx = (odom_x - self.prev_odom[0])/(self.current_time - self.last_time).to_sec()
                vy = (odom_y - self.prev_odom[1])/(self.current_time - self.last_time).to_sec()
                vth = (odom_th - self.prev_odom[2])/(self.current_time - self.last_time).to_sec()

                # abs()
                if (vx != 0 or vy != 0 or vth != 0) and self.count_steps > 1:
                    if self.start_time == 0:
                        print("vx, vy, vth, self.count_steps: ", vx, vy, vth, self.count_steps)
                        self.markers_time_list = [0, 0, 0]
                        self.start_time = time.time()

                # self.i_step += 1

                vel_mat = np.mat([[vx], [vy], [0]])
                vel_trans_mat = np.mat([[math.cos(th_map), math.sin(th_map), 0], [-math.sin(th_map), math.cos(th_map), 0], [0, 0, 1]])
                vel_trans = vel_trans_mat*vel_mat

                vx = vel_trans[0, 0]
                vy = vel_trans[1, 0]

                #self.odom_broadcaster.sendTransform(
                #    (odom_x, odom_y, 0.),
                #    odom_quat,
                #    self.current_time,
                #    "base_link",
                #    "odom"
                #)

                if self.counter % 3 == 1:
                    odom = Odometry()
                    odom.header.stamp = self.current_time
                    odom.header.frame_id = "odom"

                    odom.pose.pose = Pose(Point(odom_x, odom_y, 0.), Quaternion(*odom_quat))

                    odom.child_frame_id = "base_link"
                    odom.twist.twist = Twist(Vector3(vx, vy, 0), Vector3(0, 0, vth))

                    self.odom_pub.publish(odom)

                self.prev_odom = [odom_x, odom_y, odom_th]
                self.last_time = self.current_time

                #time_end = rospy.Time.now()
                #print("odom: ", (time_end - time_start).to_sec())

                #time_start = rospy.Time.now()

                v1 = (vx + vy + vth*0.2)/0.045
                v2 = (vx - vy - vth*0.2)/0.045
                v3 = (vx + vy - vth*0.2)/0.045
                v4 = (vx - vy + vth*0.2)/0.045

                wheel_action = np.zeros(12)

                wheel_action[11] = v1 / self.pub_rate
                wheel_action[9]  = v2 / self.pub_rate
                wheel_action[8]  = v3 / self.pub_rate
                wheel_action[10] = v4 / self.pub_rate


                self.sim.robot.arm_motor_pos = (
                    wheel_action + self.sim.robot.arm_motor_pos
                )

                #time_end = rospy.Time.now()
                #print("wheel_ctrl: ", (time_end - time_start).to_sec())

                #time_start = rospy.Time.now()

                pos_1 = self.sim.get_translation(1)
                rot_1 = self.sim.get_rotation(1)
                pose_cube_1 = Pose()
                pose_cube_1.position.x = pos_1.x
                pose_cube_1.position.y = pos_1.y
                pose_cube_1.position.z = pos_1.z
                pose_cube_1.orientation.x = rot_1.vector.x
                pose_cube_1.orientation.y = rot_1.vector.y
                pose_cube_1.orientation.z = rot_1.vector.z
                pose_cube_1.orientation.w = rot_1.scalar
                self.cube_pos_1.publish(pose_cube_1)

                pos_2 = self.sim.get_translation(3)
                rot_2 = self.sim.get_rotation(3)
                pose_cube_2 = Pose()
                pose_cube_2.position.x = pos_2.x
                pose_cube_2.position.y = pos_2.y
                pose_cube_2.position.z = pos_2.z
                pose_cube_2.orientation.x = rot_2.vector.x
                pose_cube_2.orientation.y = rot_2.vector.y
                pose_cube_2.orientation.z = rot_2.vector.z
                pose_cube_2.orientation.w = rot_2.scalar
                self.cube_pos_2.publish(pose_cube_2)

                pos_3 = self.sim.get_translation(5)
                rot_3 = self.sim.get_rotation(5)
                pose_cube_3 = Pose()
                pose_cube_3.position.x = pos_3.x
                pose_cube_3.position.y = pos_3.y
                pose_cube_3.position.z = pos_3.z
                pose_cube_3.orientation.x = rot_3.vector.x
                pose_cube_3.orientation.y = rot_3.vector.y
                pose_cube_3.orientation.z = rot_3.vector.z
                pose_cube_3.orientation.w = rot_3.scalar
                self.cube_pos_3.publish(pose_cube_3)

                pos_4 = self.sim.get_translation(7)
                rot_4 = self.sim.get_rotation(7)
                pose_cube_4 = Pose()
                pose_cube_4.position.x = pos_4.x
                pose_cube_4.position.y = pos_4.y
                pose_cube_4.position.z = pos_4.z
                pose_cube_4.orientation.x = rot_4.vector.x
                pose_cube_4.orientation.y = rot_4.vector.y
                pose_cube_4.orientation.z = rot_4.vector.z
                pose_cube_4.orientation.w = rot_4.scalar
                self.cube_pos_4.publish(pose_cube_4)

                pos_5 = self.sim.get_translation(9)
                rot_5 = self.sim.get_rotation(9)
                pose_cube_5 = Pose()
                pose_cube_5.position.x = pos_5.x
                pose_cube_5.position.y = pos_5.y
                pose_cube_5.position.z = pos_5.z
                pose_cube_5.orientation.x = rot_5.vector.x
                pose_cube_5.orientation.y = rot_5.vector.y
                pose_cube_5.orientation.z = rot_5.vector.z
                pose_cube_5.orientation.w = rot_5.scalar
                self.cube_pos_5.publish(pose_cube_5)

                pos_box = self.recive_box.translation
                rot_box = self.recive_box.rotation

                pose_target_1 = Point()
                pose_target_1.x = pos_box.x - 0.125
                pose_target_1.y = pos_box.y + 0.085
                pose_target_1.z = pos_box.z
                self.target_pos_1.publish(pose_target_1)

                pose_target_2 = Point()
                pose_target_2.x = pos_box.x
                pose_target_2.y = pos_box.y + 0.085
                pose_target_2.z = pos_box.z
                self.target_pos_2.publish(pose_target_2)

                pose_target_3 = Point()
                pose_target_3.x = pos_box.x + 0.125
                pose_target_3.y = pos_box.y + 0.085
                pose_target_3.z = pos_box.z
                self.target_pos_3.publish(pose_target_3)

                trans_box = self.recive_box.transformation

                pos_B = np.mat([[-0.13], [-0.07], [0.1], [1]])
                pos_O = np.mat([[0], [-0.07], [0.1], [1]])
                pos_X = np.mat([[0.13], [-0.07], [0.1], [1]])
                new_pos_B = trans_box * pos_B
                new_pos_O = trans_box * pos_O
                new_pos_X = trans_box * pos_X

                self.sim.set_translation(mn.Vector3(new_pos_B[0, 0], new_pos_B[1, 0], new_pos_B[2, 0]), 10)
                self.sim.set_translation(mn.Vector3(new_pos_O[0, 0], new_pos_O[1, 0], new_pos_O[2, 0]), 11)
                self.sim.set_translation(mn.Vector3(new_pos_X[0, 0], new_pos_X[1, 0], new_pos_X[2, 0]), 12)
                self.sim.set_rotation(rot_box, 10)
                self.sim.set_rotation(rot_box, 11)
                self.sim.set_rotation(rot_box, 12)

                pose_ep = Pose()
                pose_ep.position.x = self.sim.robot.sim_obj.translation.x
                pose_ep.position.y = self.sim.robot.sim_obj.translation.y
                pose_ep.position.z = self.sim.robot.sim_obj.translation.z
                pose_ep.orientation.x = self.sim.robot.sim_obj.rotation.vector.x
                pose_ep.orientation.y = self.sim.robot.sim_obj.rotation.vector.y
                pose_ep.orientation.z = self.sim.robot.sim_obj.rotation.vector.z
                pose_ep.orientation.w = self.sim.robot.sim_obj.rotation.scalar
                self.ep_pos.publish(pose_ep)

                trans_buff = self.energy_buff.transformation
                rot_buff = self.energy_buff.rotation

                pos_number_1 = np.mat([[-0.23], [0.53], [0.1], [1]])
                pos_number_2 = np.mat([[0], [0.53], [0.1], [1]])
                pos_number_3 = np.mat([[0.23], [0.53], [0.1], [1]])
                new_pos_1 = trans_buff * pos_number_1
                new_pos_2 = trans_buff * pos_number_2
                new_pos_3 = trans_buff * pos_number_3
                self.sim.set_translation(mn.Vector3(new_pos_1[0, 0], new_pos_1[1, 0], new_pos_1[2, 0]), 13)
                self.sim.set_rotation(rot_buff, 13)
                self.sim.set_translation(mn.Vector3(new_pos_2[0, 0], new_pos_2[1, 0], new_pos_2[2, 0]), 14)
                self.sim.set_rotation(rot_buff, 14)
                self.sim.set_translation(mn.Vector3(new_pos_3[0, 0], new_pos_3[1, 0], new_pos_3[2, 0]), 15)
                self.sim.set_rotation(rot_buff, 15)

                # trans_buff
                pos_buff = self.energy_buff.translation
                if self.pre_buff_w_energy == []:
                    self.pre_buff_w_energy = [pos_box, rot_box, pos_buff, rot_buff]

                # print("box, energy buff | pos, rot: ", pos_box, rot_box, trans_buff, rot_buff)

                def getNorm(mnVector3):
                    return np.linalg.norm(np.array([mnVector3.x, mnVector3.y, mnVector3.z]), 2)

                if getNorm(pos_box - self.pre_buff_w_energy[0]) \
                    + getNorm(pos_buff - self.pre_buff_w_energy[2]) > 1e-3 and self.count_steps > 40:
                    if self.is_buff_still == True:
                        print("box/buff diff: ", pos_box - self.pre_buff_w_energy[0], pos_buff - self.pre_buff_w_energy[2])
                    self.is_buff_still = False
                    for idx, marker in enumerate(self.markers_time_list):
                        if self.markers_time_list[idx] == 0.0:
                            self.markers_time_list[idx] = None
                            print(idx, None)

                self.pre_buff_w_energy = [pos_box, rot_box, pos_buff, rot_buff]

                # if self.start_time == 0:
                #     self.start_time = time.time()

                # the target_z and target_box x/y related need to set
                # target_center_z = 1.8200000524520874    # 3.400020122528076
                target_box_x_bound = 0.06
                target_box_y_bound = 0.002
                target_box_y = 0.10000018030405045
                target_box_z_bound = 0.06
                # the target_z and target_box x/y related need to set
                str = ', '.join('%s' %marker for marker in self.listExchanges)
                self.exchange_markers.publish(str)

                def within_bound(Pos, Target, bound):
                    return (Pos > Target - bound) and (Pos < Target + bound)

                # list: three random exchange [markers]
                pose_cube = [Pose(), pose_cube_1, pose_cube_2, pose_cube_3, pose_cube_4, pose_cube_5]
                pose_target = [pose_target_1, pose_target_2, pose_target_3]

                if self.pre_pose_cube == []:
                    self.pre_pose_cube = pose_cube

                def cube_pos_diff(pose, prev_pose):
                    diff_x = np.abs(pose.position.x - prev_pose.position.x)
                    diff_y = np.abs(pose.position.y - prev_pose.position.y)
                    diff_z = np.abs(pose.position.z - prev_pose.position.z)
                    diff_rot_x = np.abs(pose.orientation.x - prev_pose.orientation.x)
                    diff_rot_y = np.abs(pose.orientation.y - prev_pose.orientation.y)
                    diff_rot_z = np.abs(pose.orientation.z - prev_pose.orientation.z)
                    diff_rot_w = np.abs(pose.orientation.w - prev_pose.orientation.w)
                    if diff_x < 1e-3 and diff_y < 1e-3 and diff_z < 1e-3 and diff_rot_x < 1e-2 and diff_rot_y < 1e-2 and diff_rot_z < 1e-2 and diff_rot_w < 1e-2:
                        return True
                    else:
                        return False

                # cube_LUT = [0, 1, 3, 5, 7, 9]
                # 1. Wait for client 2. the Timer started 3. Markers 4. Foul
                # ⁣NaN, 0.0, time, None
                for idx, iTarget in enumerate(self.listExchanges):
                    # # [TODO] for test only
                    # # iTarget = cube_LUT[iTarget]
                    # if time.time() - self.start_time > idx + 1.:
                    #     pose_target[idx] = pose_cube[iTarget]   # for test
                    # #       pose_target[idx].position.z = target_center_z

                    # pose_target[idx].z
                    for iCube in range(1, 6):
                        if within_bound(pose_cube[iCube].position.y, target_box_y, target_box_y_bound) \
                                    and not self.markers_time_list[idx] and self.is_buff_still == True:
                            # print(idx, iTarget)
                            if within_bound(pose_cube[iCube].position.x, pose_target[idx].x, target_box_x_bound):
                                if within_bound(pose_cube[iCube].position.z, pose_target[idx].z, target_box_z_bound) \
                                        and cube_pos_diff(pose_cube[iCube], self.pre_pose_cube[iCube]):
                                    # print(iCube, idx, iTarget)
                                    if iCube == iTarget and self.markers_time_list[idx] == 0.0:
                                        print(time.time() - self.start_time, "\n", pose_cube[iCube], "\n", pose_target[idx])
                                        self.markers_time_list[idx] = time.time() - self.start_time
                                    elif self.markers_time_list[idx] == 0.0:
                                        self.markers_time_list[idx] = None

                str = ', '.join('%s' %marker for marker in self.markers_time_list)
                self.markers_time.publish(str)

                self.pre_pose_cube = pose_cube

                #time_end = rospy.Time.now()
                #print("positions_pub: ", (time_end - time_start).to_sec())
            #time_start = rospy.Time.now()
            self.observations = self.sim.step(action={"action": "BASE_VELOCITY", "action_args": {"base_vel": [1, 0]}})
            #time_end = rospy.Time.now()
            #print("obs_update: ", (time_end - time_start).to_sec())
        else:
            # NOTE: Here we call HabitatEvalRLEnv.step() which dispatches
            # to Env.step() or PhysicsEnv.step_physics() depending on
            # whether physics has been enabled
            self.observations = self.sim.step(self.action)

        # ------------  log sim time end  ------------
        t_sim_end = time.clock()
        with self.timing_lock:
            self.t_sim_elapsed += t_sim_end - t_sim_start
        # --------------------------------------------

        # if making video, generate frames from actions
        if self.make_video:
            self.video_frame_counter += 1
            if self.video_frame_counter == self.video_frame_period - 1:
                # NOTE: for now we only consider the case where we make videos
                # in the roam mode, for a continuous agent
                out_im_per_action = observations_to_image_for_roam(
                    self.observations,
                    {},
                    self.config.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH,
                )
                self.observations_per_episode.append(out_im_per_action)
                self.video_frame_counter = 0

        with self.command_cv:
            self.count_steps += 1

    def publish_and_step_for_eval(self):
        r"""
        Complete an episode and alert eval_episode() upon completion. Requires
        to be called after simulator reset.
        """
        # publish observations at fixed rate
        r = rospy.Rate(self.pub_rate)
        with self.enable_eval_cv:
            # wait for evaluation to be enabled
            while self.enable_eval is False:
                self.enable_eval_cv.wait()

            # publish observations and step until the episode ends
            while True:
                #time_start = rospy.Time.now()
                self.publish_sensor_observations()
                #time_end = rospy.Time.now()
                #print("obs_time: ", (time_end - time_start).to_sec())

                #time_start = rospy.Time.now()
                self.step()
                #time_end = rospy.Time.now()
                #print("step_time: ", (time_end - time_start).to_sec())
                #print("#############################################################")
                r.sleep()

            # now the episode is done, disable evaluation and alert eval_episode()
            self.enable_eval = False
            self.enable_eval_cv.notify()

    def publish_and_step_for_roam(self):
        r"""
        Let an agent roam within a scene until shutdown. Requires to be called
        1) after simulator reset, 2) shutdown_lock has not yet been acquired by
        the current thread.
        """
        # publish observations at fixed rate
        r = rospy.Rate(self.pub_rate)
        with self.enable_eval_cv:
            # wait for evaluation to be enabled
            while self.enable_eval is False:
                self.enable_eval_cv.wait()

            # publish observations and step until shutdown
            while True:
                with self.shutdown_lock:
                    if self.shutdown:
                        break
                #time_start = rospy.Time.now()
                self.publish_sensor_observations()
                #time_end = rospy.Time.now()
                #print("obs_time: ", (time_end - time_start).to_sec())

                #time_start = rospy.Time.now()
                self.step()
                #time_end = rospy.Time.now()
                #print("step_time: ", (time_end - time_start).to_sec())
                #print("#############################################################")
                r.sleep()

            # disable evaluation
            self.enable_eval = False

    def callback(self, cmd_msg):
        r"""
        Takes in a command from an agent and alert the simulator to enact
        it.
        :param cmd_msg: Either a velocity command or an action command.
        """
        # unpack agent action from ROS message, and send the action
        # to the simulator
        with self.command_cv:
            if self.use_continuous_agent:
                # set linear + angular velocity
                self.linear_vel = np.array(
                    [-cmd_msg.linear.y, 0.0, -cmd_msg.linear.x]
                )
                self.angular_vel = np.array([0.0, cmd_msg.angular.z, 0.0])
                #self.arm_action_cfg = [cmd_msg.angular.x, cmd_msg.angular.y]
                #self.switch = cmd_msg.linear.z
            else:
                # get the action
                self.action = cmd_msg.data

            # set action publish flag and notify
            self.new_command_published = True
            self.command_cv.notify()
    def callback_gps(self, gps_data):
            self.goal_position=np.matrix([[gps_data.goal.target_pose.pose.position.x], [gps_data.goal.target_pose.pose.position.y], [gps_data.goal.target_pose.pose.position.z], [1.0]] )
            self.goal_rotation=gps_data.goal.target_pose.pose.orientation

    def tf_callback(self, cmd_msg):
        if cmd_msg.transforms[0].child_frame_id == 'tracker_LHR_A655F116' or cmd_msg.transforms[0].child_frame_id == 'tracker_LHR_28F9E075' or  cmd_msg.transforms[0].child_frame_id == 'tracker_LHR_79DF4EBF' or cmd_msg.transforms[0].child_frame_id == 'tracker_LHR_38294716':
            translation = cmd_msg.transforms[0].transform.translation
            rotation = cmd_msg.transforms[0].transform.rotation

            r, p, y = self.quat2ang(rotation.x, rotation.y, rotation.z, rotation.w)
            p += 1.57
            if r < 0:
                p = -p
            p += 1.57
            w, x, y, z = self.ang2quat(0, p, 0)

            self.sim.robot.sim_obj.rotation = mn.Quaternion(mn.Vector3(x, y, z), w)

            trans_mat = np.mat([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
            trans = np.mat([[translation.x], [translation.y], [translation.z]])
            new_trans = trans_mat * trans

            new_trans = new_trans.T + self.tf_init_trans

            trans_mat = np.mat([[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, -math.cos(p)]])
            x_trans = np.mat([[0.125], [0], [0]])
            x_trans = trans_mat * x_trans

            new_trans += x_trans.T

            self.sim.robot.sim_obj.translation = mn.Vector3(new_trans[0, 0], self.sim.robot.sim_obj.translation[1], new_trans[0, 2])

    def callback1(self, cmd_msg):
        with self.command_cv:
            self.switch = cmd_msg.x

    def callback2(self, cmd_msg):
        with self.command_cv:
            self.arm_action_cfg = [cmd_msg.position.x*1000, cmd_msg.position.y*1000]

    def callback3(self, cmd_msg):
        with self.command_cv:
            self.init_move_pos = [cmd_msg.linear.x, -cmd_msg.linear.y, cmd_msg.angular.z]

    def quat2ang(self, x, y ,z, w):
        r = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        p = math.asin(2 * (w * y - z * x))
        y = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return r, p, y
    
    def ang2quat(self, r, p, y):
        sinp = math.sin(p/2)
        siny = math.sin(y/2)
        sinr = math.sin(r/2)

        cosp = math.cos(p/2)
        cosy = math.cos(y/2)
        cosr = math.cos(r/2)

        w = cosr*cosp*cosy + sinr*sinp*siny
        x = sinr*cosp*cosy - cosr*sinp*siny
        y = cosr*sinp*cosy + sinr*cosp*siny
        z = cosr*cosp*siny - sinr*sinp*cosy

        return w, x, y, z

    def simulate(self):
        r"""
        An infinite loop where the env node 1) keeps evaluating the next
        episode in its RL environment, if an EvalEpisode request is given;
        or 2) let the agent roam freely in one episode.
        Breaks upon receiving shutdown command.
        """
        # iterate over episodes
        while True:
            try:
                # reset the env
                self.reset()
                with self.shutdown_lock:
                    # if shutdown service called, exit
                    if self.shutdown:
                        rospy.signal_shutdown("received request to shut down")
                        break
                with self.enable_reset_cv:
                    if self.enable_roam:
                        self.publish_and_step_for_roam()
                    else:
                        # otherwise, evaluate the episode
                        self.publish_and_step_for_eval()
            except StopIteration:
                # set enable_reset and enable_eval to False, so the
                # env node can evaluate again in the future
                with self.enable_reset_cv:
                    self.enable_reset = False
                with self.enable_eval_cv:
                    self.all_episodes_evaluated = True
                    self.enable_eval = False
                    self.enable_eval_cv.notify()

    def on_exit_generate_video(self):
        r"""
        Make video of the current episode, if video production is turned
        on.
        """
        if self.make_video:
            generate_video(
                video_option=self.config.VIDEO_OPTION,
                video_dir=self.config.VIDEO_DIR,
                images=self.observations_per_episode,
                episode_id="fake_episode_id",
                scene_id="fake_scene_id",
                agent_seed=0,
                checkpoint_idx=0,
                metrics={},
                tb_writer=None,
            )


def main():
    # parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--node-name", type=str, default="env_node")
    parser.add_argument(
        "--task-config", type=str, default="configs/pointnav_d_orignal.yaml"
    )
    parser.add_argument("--enable-physics-sim", default=False, action="store_true")
    parser.add_argument("--use-continuous-agent", default=False, action="store_true")
    parser.add_argument(
        "--sensor-pub-rate",
        type=float,
        default=30.0,
    )
    parser.add_argument(
        "--round",
        type=int,
        default=0,
    )
    args = parser.parse_args()

    # initialize the env node
    env_node = HabitatEnvNode(
        node_name=args.node_name,
        config_paths=args.task_config,
        enable_physics_sim=args.enable_physics_sim,
        use_continuous_agent=args.use_continuous_agent,
        pub_rate=args.sensor_pub_rate,
        round=args.round,
    )

    # run simulations
    env_node.simulate()


if __name__ == "__main__":
    main()

#!/usr/bin/env python

# Import modules
import numpy as np
import random
from collections import Counter
import sklearn
from sklearn.preprocessing import LabelEncoder
import pickle
from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker
from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *

import rospy
import tf
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from pr2_robot.srv import *
from rospy_message_converter import message_converter
import yaml


# Helper function to get surface normals
def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

# Helper function to create a yaml friendly dictionary from ROS messages
def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    yaml_dict = {}
    yaml_dict["test_scene_num"] = test_scene_num.data
    yaml_dict["arm_name"]  = arm_name.data
    yaml_dict["object_name"] = object_name.data
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(pick_pose)
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(place_pose)
    return yaml_dict

# Helper function to output to yaml file
def send_to_yaml(yaml_filename, dict_list):
    data_dict = {"object_list": dict_list}
    with open(yaml_filename, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)

    
def get_feature(cloud):
    chists = compute_color_histograms(cloud, using_hsv=True) 
    normals = get_normals(cloud)
    nhists = compute_normal_histograms(normals)
    feature = np.concatenate((chists, nhists))
    return feature
    
    
def find_clusters(objects_cloud, tolerance=0.05, min_size=20, max_size=20000):
    """Euclidean Clustering"""
    # Remove color and cluster by distance only.
    white_cloud = XYZRGB_to_XYZ(objects_cloud)
    tree = white_cloud.make_kdtree()
    ec = white_cloud.make_EuclideanClusterExtraction()
    ec.set_ClusterTolerance(tolerance)  # meters: 0.05m = 5cm
    ec.set_MinClusterSize(min_size)
    ec.set_MaxClusterSize(max_size)
    ec.set_SearchMethod(tree)
    cluster_indices = ec.Extract()
    print len(cluster_indices), "clusters"
    return cluster_indices


def create_cluster_point_cloud(objects_cloud, cluster_indices):
    """Creates a colored Cluster-Mask Point Cloud to visualize each cluster separately.
    """
    cluster_color = get_color_list(len(cluster_indices))
    color_cluster_point_list = []
    for i, indices in enumerate(cluster_indices):
        new_color = rgb_to_float(cluster_color[i])
        # objects_cloud[index] is a length four tuple: (x,y,z,rgb_float)
        points = [list(objects_cloud[index])[:3] + [new_color] for index in indices]
        print "{} has {} points".format(i, len(points))
        color_cluster_point_list.extend(points)
    cluster_cloud = pcl.PointCloud_PointXYZRGB()
    cluster_cloud.from_list(color_cluster_point_list)
    return cluster_cloud

    
# Create publishers
PCL_OBJECTS = "/pcl_objects"
PCL_TABLE = "/pcl_table"
PCL_CLUSTER = "/pcl_cluster"
OBJECT_MARKERS = "/object_markers"
DETECTED_OBJECTS = "/detected_objects"
OBJECT_LIST = "/object_list"
DROPBOXES = "/dropbox"
TEST_SCENE_NUM = rospy.get_param("/test_scene_num")
pubs = {
    PCL_OBJECTS: rospy.Publisher(PCL_OBJECTS, PointCloud2, queue_size=1),
    PCL_TABLE: rospy.Publisher(PCL_TABLE, PointCloud2, queue_size=1),
    PCL_CLUSTER: rospy.Publisher(PCL_CLUSTER, PointCloud2, queue_size=1),
    OBJECT_MARKERS: rospy.Publisher(OBJECT_MARKERS, Marker, queue_size=1),
    DETECTED_OBJECTS: rospy.Publisher(DETECTED_OBJECTS, DetectedObjectsArray, queue_size=1),
}

        
class ObjectFinder(object):
    def __init__(self, filter_noise=False):        
        model = pickle.load(open('model.sav', 'rb'))
        self.clf = model['classifier']
        self.encoder = LabelEncoder()
        self.encoder.classes_ = model['classes']
        self.scaler = model['scaler']
        
        self.filter_noise = filter_noise
        
        self.last_cloud_objects = None
        
       
    def pcl_callback(self, pcl_msg):
        print "=================="
        print "PCL CALLBACK START"

        # Convert ROS msg to PCL data
        cloud = ros_to_pcl(pcl_msg)
        
    
        # Statistical Outlier Filtering
        STAT_MEAN_K = rospy.get_param('/filter/stat_mean_k', 20)
        print "'/filter/stat_mean_k' = {}".format(STAT_MEAN_K)
        STAT_THRESH = rospy.get_param('/filter/stat_thresh', 0.005)  # 0.2 is already much better than 1.0
        print "'/filter/stat_thresh' = {}".format(STAT_THRESH)
        if STAT_MEAN_K > 0 and STAT_THRESH > 0.0:
            outlier_filter = cloud.make_statistical_outlier_filter()
            outlier_filter.set_mean_k(STAT_MEAN_K)
            outlier_filter.set_std_dev_mul_thresh(STAT_THRESH)
            cloud_filtered = outlier_filter.filter()
        else:
            print "  - Skipping statistical outlier filtering"
        
        # Voxel Grid Downsampling
        LEAF_SIZE = rospy.get_param('/filter/voxel_leaf', 0.007)
        print "'/filter/voxel_leaf' = {}".format(LEAF_SIZE)
        if LEAF_SIZE > 0.0:
            vox = cloud_filtered.make_voxel_grid_filter()
            vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
            cloud_filtered = vox.filter()
        else:
            print "  - Skipping voxel grid downsampling"
            
        

        # PassThrough Filter - X
        passthrough = cloud_filtered.make_passthrough_filter()
        filter_axis = 'x'
        passthrough.set_filter_field_name(filter_axis)
        axis_min = 0.35
        axis_max = 1.00
        passthrough.set_filter_limits(axis_min, axis_max)
        cloud_filtered = passthrough.filter()

        # PassThrough Filter - Y
        passthrough = cloud_filtered.make_passthrough_filter()
        filter_axis = 'y'
        passthrough.set_filter_field_name(filter_axis)
        axis_min = -.7
        axis_max = 0.7
        passthrough.set_filter_limits(axis_min, axis_max)
        cloud_filtered = passthrough.filter()

        # PassThrough Filter - Z
        passthrough = cloud_filtered.make_passthrough_filter()
        filter_axis = 'z'
        passthrough.set_filter_field_name(filter_axis)
        axis_min = 0.6
        axis_max = 1.5
        passthrough.set_filter_limits(axis_min, axis_max)
        cloud_filtered = passthrough.filter()

        # RANSAC Plane Segmentation
        seg = cloud_filtered.make_segmenter()
        seg.set_model_type(pcl.SACMODEL_PLANE)
        seg.set_method_type(pcl.SAC_RANSAC)
        max_distance = 0.01
        seg.set_distance_threshold(max_distance)
        inliers, coefficients = seg.segment()

        # Extract inliers and outliers
        extracted_inliers = cloud_filtered.extract(inliers, negative=False)
        extracted_objects = cloud_filtered.extract(inliers, negative=True)

        # Statistical Outlier Filtering
        # Do again to remove remaining bits of the table
        STAT_MEAN_K_2 = rospy.get_param('/filter/stat_mean_k_2', 30)
        print "'/filter/stat_mean_k_2' = {}".format(STAT_MEAN_K_2)
        STAT_THRESH_2 = rospy.get_param('/filter/stat_thresh_2', 0.4)  # 0.2 is already much better than 1.0
        print "'/filter/stat_thresh_2' = {}".format(STAT_THRESH_2)
        if STAT_MEAN_K_2 > 0 and STAT_THRESH_2 > 0.0:
            outlier_filter = extracted_objects.make_statistical_outlier_filter()
            outlier_filter.set_mean_k(STAT_MEAN_K_2)
            outlier_filter.set_std_dev_mul_thresh(STAT_THRESH_2)
            extracted_objects = outlier_filter.filter()
        else:
            print "  - Skipping 2ND statistical outlier filtering"
        
            
        if self.last_cloud_objects:
            new_cloud = pcl.PointCloud_PointXYZRGB()
            merged_list = extracted_objects.to_list() + self.last_cloud_objects.to_list()
            new_cloud.from_list(merged_list)
            extracted_objects = new_cloud
            print "New cloud merged with old"
        
        #self.last_cloud_objects = extracted_objects
        
        #####################
        ## Finding clusters
        #####################
        cluster_indices = find_clusters(extracted_objects)        
        cluster_cloud = create_cluster_point_cloud(extracted_objects, cluster_indices)

        #####################
        ## Publish transformed data
        #####################
        # Convert PCL data to ROS messages
        ros_cloud_table = pcl_to_ros(extracted_inliers)
        ros_cloud_objects = pcl_to_ros(extracted_objects)
        ros_cluster_cloud = pcl_to_ros(cluster_cloud)

        # Publish ROS messages
        pubs[PCL_OBJECTS].publish(ros_cloud_objects)
        pubs[PCL_TABLE].publish(ros_cloud_table)
        pubs[PCL_CLUSTER].publish(ros_cluster_cloud)

        #####################
        ## Classify the clusters (Name the objects)
        #####################
        detected_objects_labels = []
        detected_objects = []
        # Publish the list of detected objects
        
        counter_size = rospy.get_param('/predict/iterations', 50)
        sample_size = rospy.get_param('/predict/sample_size', 30)
        print "'/predict/sample_size' = {}".format(sample_size)
        print "'/predict/iterations' = {}".format(counter_size)
        for index, pts_list in enumerate(cluster_indices):
            # Grab the points for the cluster from the extracted outliers (cloud_objects)
            object_cloud = extracted_objects.extract(pts_list)
            c_size = object_cloud.size
            
            mode_list = []
            for i in range(counter_size):
                indices = random.sample(range(c_size), min(sample_size, c_size))
                subset_cloud = object_cloud.extract(indices)
                ros_cloud = pcl_to_ros(subset_cloud)  # to sensor_msgs.msg._PointCloud2.PointCloud2

                # Extract histogram features
                feature = get_feature(ros_cloud)

                # Make the prediction, retrieve the label for the result
                # and add it to detected_objects_labels list
                prediction = self.clf.predict(self.scaler.transform(feature.reshape(1,-1)))
                mode_list.append(prediction[0])
            print Counter(mode_list).most_common(2)
            prediction = Counter(mode_list).most_common(1)[0][0]
            label = self.encoder.inverse_transform(prediction)
            detected_objects_labels.append(label)

            # Add the detected object to the list of detected objects.
            do = DetectedObject()
            do.label = label
            do.cloud = ros_cloud
            detected_objects.append(do)

            # Publish a label into RViz
            label_pos = list(extracted_objects[pts_list[0]])
            label_pos[2] += .4
            pubs[OBJECT_MARKERS].publish(make_label(label, label_pos, index))

        rospy.loginfo('Detected {} objects: {}'.format(len(detected_objects_labels), detected_objects_labels))

        # Publish the list of detected objects
        pubs[DETECTED_OBJECTS].publish(detected_objects)

        write_yaml(detected_objects)
        
        """
        # Suggested location for where to invoke your pr2_mover() function within pcl_callback()
        # Could add some logic to determine whether or not your object detections are robust
        # before calling pr2_mover()
        try:
            pr2_mover(detected_objects)
        except rospy.ROSInterruptException:
            pass
        """
        print "------------------"
            
def write_yaml(detected_objects):
    labels = [item.label for item in detected_objects]
    centroids = [get_cloud_center(item.cloud) for item in detected_objects]
    
    # Get/Read parameters
    pick_list = rospy.get_param(OBJECT_LIST)
    
    # Dropbox info organized by group
    dropboxes = {_['group']: (_['name'],_['position']) for _ in rospy.get_param(DROPBOXES)}
    
    # Messages to hold data
    test_scene_num = Int32()
    test_scene_num.data = TEST_SCENE_NUM
    object_name = String()
    arm_name = String()
    pick_pose = Pose()
    place_pose = Pose()
    
    dict_list = []
    for list_item in pick_list:
        name, group = list_item['name'], list_item['group']
        # Continue if object not seen.
        try:
            centroid = centroids[labels.index(name)]
        except ValueError:
            continue
        
        object_name.data = name
        arm_name.data = dropboxes[group][0]
        pick_pose.position.x, \
        pick_pose.position.y, \
        pick_pose.position.z = [np.asscalar(_) for _ in centroid]
        place_pose.position.x, \
        place_pose.position.y, \
        place_pose.position.z = dropboxes[group][1]
        
        dict_list.append(make_yaml_dict(test_scene_num, 
                                        arm_name, 
                                        object_name, 
                                        pick_pose, 
                                        place_pose))
    
    send_to_yaml("output_{}.yaml".format(TEST_SCENE_NUM), dict_list)


def get_cloud_center(cloud):
    # Convert cloud points to array and return mean of x,y,z.
    return np.mean(ros_to_pcl(cloud).to_array(), axis=0)[:3]

# function to load parameters and request PickPlace service
def pr2_mover(detected_objects):
    pass
    # Get/Read parameters
    

    # Parse parameters into individual variables
   

    # TODO: Rotate PR2 in place to capture side tables for the collision map
    

    # TODO: Loop through the pick list

        # TODO: Get the PointCloud for a given object and obtain it's centroid

        # TODO: Create 'place_pose' for the object

        # TODO: Assign the arm to be used for pick_place

        # TODO: Create a list of dictionaries (made with make_yaml_dict()) for later output to yaml format

    """
        # Wait for 'pick_place_routine' service to come up
        rospy.wait_for_service('pick_place_routine')

        try:
            pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)

            # TODO: Insert your message variables to be sent as a service request
            resp = pick_place_routine(TEST_SCENE_NUM, OBJECT_NAME, WHICH_ARM, PICK_POSE, PLACE_POSE)

            print ("Response: ",resp.success)

        except rospy.ServiceException, e:
            print "Service call failed: %s"%e
    """
    # TODO: Output your request parameters into output yaml file



if __name__ == '__main__':
    # NOTE: anonymous adds a random number to name to make it unique
    rospy.init_node('J_perception_pipeline', anonymous=True)

    # Initialize color_list
    get_color_list.color_list = []
    obj_finder = ObjectFinder()
    
    rospy.Subscriber("/pr2/world/points", PointCloud2, obj_finder.pcl_callback)

    rospy.spin()  # Needed for subscribers

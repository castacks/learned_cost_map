#!/usr/bin/python3
import rospy
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import rosbag
import cv2
from mpl_toolkits.axes_grid1 import make_axes_locatable

from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

from std_msgs.msg import Header
from nav_msgs.msg import OccupancyGrid, MapMetaData, Odometry
import time

class CostmapViz(object):
    def __init__(
                    self,
                    save_dir,
                    experiment_dir,
                    learned_costmap_topic,
                    baseline_costmap_topic,
                    rgbmap_topic,
                    odom_topic,
                    image_topic,
                    save_every,
                    vertical
                ):

        self.save_dir = save_dir.rstrip('/')
        self.experiment_dir = experiment_dir.rstrip('/')
        self.learned_costmap_topic = learned_costmap_topic
        self.baseline_costmap_topic = baseline_costmap_topic
        self.rgbmap_topic = rgbmap_topic
        self.odom_topic = odom_topic
        self.image_topic = image_topic
        self.save_every = save_every
        self.vertical = vertical
        self.figcnt = 0
        self.cvbridge = CvBridge()

    def make_figures(self):
        save_fp = os.path.join(self.save_dir, os.path.basename(self.experiment_dir))
        print('generating figures from experiment {} and saving to {}'.format(self.experiment_dir, save_fp))

        if os.path.exists(save_fp):
            inp = input('{} exists. Overwrite? [Y/n]'.format(save_fp))
            if inp == 'n':
                exit(0)
        else:
            os.mkdir(save_fp)

        experiment_fps = os.listdir(self.experiment_dir)
        for ex_bn in experiment_fps:
            self.figcnt = 0
            efp = os.path.join(self.experiment_dir, ex_bn)
            save_exp_fp = os.path.join(save_fp, ex_bn)
            if not os.path.exists(save_exp_fp):
                os.mkdir(save_exp_fp)

            bag_fps = np.array(os.listdir(efp))
            ctimes = np.array([os.path.getctime(os.path.join(efp, x)) for x in bag_fps])
            bag_fps = bag_fps[np.argsort(ctimes)]

            for i, bag_fp in enumerate(bag_fps):
                print('processing {}/{} ({})'.format(i+1, len(bag_fps), bag_fp))
                self.parse_bag(os.path.join(efp, bag_fp), save_exp_fp)

    def parse_bag(self, bag_fp, save_fp):
        last_save_time = -1e10
        current_time = -1e10
        topics = [
                    self.learned_costmap_topic,
                    self.baseline_costmap_topic,
                    self.rgbmap_topic,
                    self.odom_topic,
                    self.image_topic,
                    ]
        current_msgs = {k:None for k in topics}
        bag = rosbag.Bag(os.path.join(self.experiment_dir, bag_fp), 'r')
        for topic, msg, t in bag.read_messages(topics=topics):
            current_time = t.to_sec()
            current_msgs[topic] = msg
            if current_time - last_save_time > self.save_every:
                last_save_time = current_time
                if all([v is not None for v in current_msgs.values()]):
                    print('process time {}'.format(current_time))
                    save_fig_fp = os.path.join(save_fp, '{:05d}.png'.format(self.figcnt))
                    self.save_figure(current_msgs, save_fig_fp)
                    self.figcnt += 1
                else:
                    print('waiting for these topics:\n{}'.format([k for k,v in current_msgs.items() if v is None]))

    def save_figure(self, msgs, fp):
        front_image = self.cvbridge.imgmsg_to_cv2(msgs[self.image_topic], "rgb8")
        rgbmap = self.cvbridge.imgmsg_to_cv2(msgs[self.rgbmap_topic], "rgb8")
        lmsg = msgs[self.learned_costmap_topic]
        learned_map = np.reshape(np.array(list(lmsg.data))/100.0, (lmsg.info.width, lmsg.info.height)).T

        bmsg = msgs[self.baseline_costmap_topic]
        baseline_map = np.reshape(np.array(list(bmsg.data))/100.0, (bmsg.info.width, bmsg.info.height)).T

        #small hack - we're assuming that the local and global maps have the same origin.
        #also assuming that the robot is the center of the baseline map
        odom_msg = msgs[self.odom_topic].pose.pose
        qw = odom_msg.orientation.w
        qx = odom_msg.orientation.x
        qy = odom_msg.orientation.y
        qz = odom_msg.orientation.z
        yaw = np.arctan2(2*(qw*qz + qx*qy), 1 - 2 * (qy*qy + qz*qz))

        baseline_map_img = baseline_map.astype(np.uint8) * 100
        img_center = (baseline_map_img.shape[0]//2, baseline_map_img.shape[1]//2)
        R = cv2.getRotationMatrix2D(center=img_center, angle=-yaw * (180./np.pi), scale=1)
        #now crop to the localmap dim
        baseline_map_img = cv2.warpAffine(baseline_map_img, R, baseline_map_img.shape)
        baseline_map_crop = baseline_map_img.astype(np.float32)

        learned_min_x = int(lmsg.info.origin.position.x / bmsg.info.resolution)
        learned_min_y = int(lmsg.info.origin.position.y / bmsg.info.resolution)
        learned_max_x = learned_min_x + int(lmsg.info.width * (lmsg.info.resolution/bmsg.info.resolution))
        learned_max_y = learned_min_y + int(lmsg.info.height * (lmsg.info.resolution/bmsg.info.resolution))
        baseline_map_crop = baseline_map_img[img_center[0]+learned_min_x:img_center[0]+learned_max_x, img_center[1]+learned_min_y:img_center[1]+learned_max_y]
        rescale_shape = (
            int(baseline_map_crop.shape[0] * (bmsg.info.resolution/lmsg.info.resolution)),
            int(baseline_map_crop.shape[1] * (bmsg.info.resolution/lmsg.info.resolution)),
        )

        baseline_map_crop = cv2.resize(baseline_map_crop, rescale_shape, interpolation=cv2.INTER_NEAREST) / 100

        mosaic = """
                  A
                  B
                  B
                  C
                  C
                  D
                  D
                  E
                  E
                 """ if self.vertical else """AABCDE"""

        fig = plt.figure(constrained_layout=True, figsize=(3, 15) if self.vertical else (18, 3))
        axs = fig.subplot_mosaic(mosaic)

        map_extent = (0., bmsg.info.width*bmsg.info.resolution, 0., bmsg.info.height*bmsg.info.resolution)
        ldy1 = lmsg.info.origin.position.x
        ldy2 = ldy1 + lmsg.info.resolution * lmsg.info.width
        ldx1 = lmsg.info.origin.position.y
        ldx2 = ldx1 + lmsg.info.resolution * lmsg.info.height

        corners = np.array([
            [map_extent[1]/2 + ldx1 * np.cos(yaw) - ldy1 * np.sin(yaw), map_extent[3]/2 + ldx1 * np.sin(yaw) + ldy1 * np.cos(yaw)],
            [map_extent[1]/2 + ldx1 * np.cos(yaw) - ldy2 * np.sin(yaw), map_extent[3]/2 + ldx1 * np.sin(yaw) + ldy2 * np.cos(yaw)],
            [map_extent[1]/2 + ldx2 * np.cos(yaw) - ldy2 * np.sin(yaw), map_extent[3]/2 + ldx2 * np.sin(yaw) + ldy2 * np.cos(yaw)],
            [map_extent[1]/2 + ldx2 * np.cos(yaw) - ldy1 * np.sin(yaw), map_extent[3]/2 + ldx2 * np.sin(yaw) + ldy1 * np.cos(yaw)],
            [map_extent[1]/2 + ldx1 * np.cos(yaw) - ldy1 * np.sin(yaw), map_extent[3]/2 + ldx1 * np.sin(yaw) + ldy1 * np.cos(yaw)],
        ])

        for i in range(4):
            axs['E'].plot(corners[[i, i+1], 0], corners[[i, i+1], 1], c='r', linewidth=2)

        axs['E'].arrow(map_extent[1]/2, map_extent[3]/2, -3*np.sin(yaw), 3*np.cos(yaw), head_width=0.9, color='r')

        axs['A'].imshow(front_image)
        axs['B'].imshow(rgbmap[:, ::-1], origin='lower')
        axs['C'].imshow(learned_map[:, ::-1], origin='lower', cmap='plasma', vmin=0., vmax=1.0)
        axs['D'].imshow(baseline_map_crop[:, ::-1], origin='lower', cmap='plasma', vmin=0., vmax=1.0)
        axs['E'].imshow(baseline_map[:, ::-1], origin='lower', cmap='plasma', vmin=0., vmax=1.0, extent=map_extent)

        for ax in axs.values():
            ax.set_xticks([])
            ax.set_yticks([])

#        plt.show()
        plt.savefig(fp, dpi=300, bbox_inches='tight')
        plt.close()

    def save_costmap_figs(self, count):
        if self.learned_costmap is None:
            print("Haven't obtained learned costmap")
            return

        # axs.clear()

        # divider = make_axes_locatable(axs)
        # cax = divider.append_axes('right', size='5%', pad=0.05)
        # fig.colorbar(self.learned_costmap, cax=cax, orientation='vertical')
        learned_costmap_dir = f"/home/mateo/corl_plots/data_collect1/learned_costmaps/frame_{count:08}.png"
        front_facing_dir = f"/home/mateo/corl_plots/data_collect1/front_facing/frame_{count:08}.png"
        rgb_map_dir = f"/home/mateo/corl_plots/data_collect1/rgb_maps/frame_{count:08}.png"

        plt.clf()
        plt.title("Learned Costmap")
        plt.imshow(self.learned_costmap, origin="lower", vmin=0.0, vmax=1.0)
        plt.colorbar()
        plt.savefig(learned_costmap_dir, dpi=300, bbox_inches="tight")
        plt.clf()


        plt.title("Front-Facing Image")
        plt.imshow(self.front_facing)
        plt.axis('off')
        plt.savefig(front_facing_dir, dpi=300, bbox_inches="tight")
        plt.clf()

        plt.title("RGB Map")
        plt.imshow(self.rgbmap_inflate, origin="lower")
        plt.colorbar()
        plt.savefig(rgb_map_dir, dpi=300, bbox_inches="tight")
        plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, required=True, help='directory to save results in. We will create a dir under this with the same name as the experiment dir')
    parser.add_argument('--experiment_dir', type=str, required=True)
    parser.add_argument('--learned_costmap_topic', type=str, required=False, default='/learned_costmap', help='topic to read learned costmap from')
    parser.add_argument('--baseline_costmap_topic', type=str, required=False, default='/local_cost_map_final_occupancy_grid', help='topic to read learned costmap from')
    parser.add_argument('--rgbmap_topic', type=str, required=False, default='/local_rgb_map_inflate')
    parser.add_argument('--odom_topic', type=str, required=False, default='/integrated_to_init')
    parser.add_argument('--image_topic', type=str, required=False, default='/multisense/left/image_rect_color')
    parser.add_argument('--vertical', type=int, required=False, default=0, help='make the plots vertical or not')
    parser.add_argument('--save_every', type=float, required=False, default=5., help='save a figure every x seconds')
    args = parser.parse_args()

    vertical = args.vertical > 0

    node = CostmapViz(save_dir=args.save_dir, experiment_dir=args.experiment_dir, learned_costmap_topic=args.learned_costmap_topic, baseline_costmap_topic=args.baseline_costmap_topic, rgbmap_topic=args.rgbmap_topic, odom_topic=args.odom_topic, image_topic=args.image_topic, save_every=args.save_every, vertical=vertical)
    node.make_figures()

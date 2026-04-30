#!/bin/bash

sudo chmod 666 /dev/ttyACM0
sudo chmod 666 /dev/ttyACM1


lerobot-teleoperate \
  --robot.type=so101_follower --robot.port='/dev/ttyACM0' --robot.id=my_awesome_follower_arm \
  --teleop.type=so101_leader --teleop.port='/dev/ttyACM1' --teleop.id=my_awesome_leader_arm \
  --robot.cameras="{ front: {type: opencv, index_or_path: 2, width: 480, height: 640, fps: 30, rotation: -90}}" \
  --display_data=false

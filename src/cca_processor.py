from src.neural_analysis import *
from src.cort_processor import *
from src.wiener_filter import *
from src.folder_handler import *
from src.tdt_support import *
from src.decoders import *

class CCAProcessor:
    def __init__(self, cp1, cp2, limbfoot_angle=1):
        self.cp1 = cp1
        self.cp1.decode_angles()
        self.cp1.get_gait_indices()
        self.cp2 = cp2
        self.cp2.decode_angles()
        self.cp2.get_gait_indices()
        self.limbfoot_angle=limbfoot_angle

        print(self.check_same_kinematics())

    def check_same_kinematics(self):
        if self.cp1.data['angle_names'] == self.cp2.data['angle_names']:
            output = 'should be good to align'
            return output
        else:
            output = 'kinematics are different'
            return output

    def process_and_align_kinematics(self):
        cp1_gait_x, cp1_gait_y = self.cp1.remove_bad_gaits()
        cp2_gait_x, cp2_gait_y = self.cp2.remove_bad_gaits()
        return cp1_gait_x, cp2_gait_x
        

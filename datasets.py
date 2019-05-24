
from utils import *
import torch.utils.data
from sklearn.model_selection import train_test_split
#from ipypb import ipb
from tqdm import tqdm

rseed = 42

class pamap2Dataset(torch.utils.data.Dataset):
    
    """
        
    """
    def __init__(self,subjects,fold='train',
                 win_size=256,step=128,
                 num_classes = 4,
                 test_subject = 6,
                 val_rate = 0.3):
        """
        """
        self.data = []
        self.labels = []
        self.test_data = []
        self.test_labels = []
        self.win_size = win_size
        self.step = step
        
        self.fold = fold
        
        
        for i in tqdm(range(len(subjects))):
            #data = []
            #labels = []
            subject = subjects[i]
            for window in sliding_window(subject[useful_columns],win_size = self.win_size,
                                         step=self.step):
                if i != test_subject:
                    self.data.append(window[data_columns])
                    self.labels.append(major_filter(window['activityID']))
                else:
                    self.test_data.append(window[data_columns])
                    self.test_labels.append(major_filter(window['activityID']))
        self.data = np.vstack(self.data)
        self.data = self.data.reshape([-1,self.win_size,len(data_columns)]).astype('float32')
        self.labels = np.hstack(self.labels)
        
        
        self.test_data = np.vstack(self.test_data)
        self.test_data = self.test_data.reshape([-1,self.win_size,len(data_columns)]).astype('float32')
        self.test_labels = np.hstack(self.test_labels)
        self.test_labels = one_hot(self.test_labels,num_classes).astype('float32')
        
        self.labels = one_hot(self.labels,num_classes).astype('float32')
        
        self.total_size = len(self.data)
        
        indxs = train_test_split(np.arange(self.total_size),test_size=val_rate,random_state=rseed)
        self.train_idxs = indxs[0]
        self.val_idxs = indxs[1]
        
        
        ## actions in original data are sequential,
        ## e.g. subject makeas action1, then action2 ... action3
        #self.val_rate = val_rate
        #self.val_size = int(self.val_rate*self.total_size)

        #self.train_rate = 1-self.val_rate
        #self.train_size = int(self.train_rate*self.total_size)
        
        self.train_data = self.data[self.train_idxs]
        self.train_labels = self.labels[self.train_idxs]
        self.val_data = self.data[self.val_idxs]
        self.val_labels = self.labels[self.val_idxs]
        
        
    def __len__(self,):
        """
        """
        if self.fold == 'train':
            return len(self.train_data)
        elif self.fold == 'val':
            return len(self.val_data)
        elif self.fold == 'test':
            return len(self.test_data)
    
    def __getitem__(self, index):
        """
        """
        if self.fold == 'train':
            data = self.train_data[index]
            label =  self.train_labels[index]
        elif self.fold == 'val':
            data = self.val_data[index]
            label =  self.val_labels[index]
        elif self.fold == 'test':
            data = self.test_data[index]
            label =  self.test_labels[index]
        return data,label

#### predefined
    
num2label = {1: 'lying',
               2: 'sitting',
               3: 'standing',
               4: 'walking',
               5: 'running',
               6: 'cycling',
               7: 'Nordic walking',
               9: 'watching TV',
               10: 'computer work',
               11: 'car driving',
               12: 'ascending stairs',
               13: 'descending stairs',
               16: 'vacuum cleaning',
               17: 'ironing',
               18: 'folding laundry',
               19: 'house cleaning',
               20: 'playing soccer',
               24: 'rope jumping',
                0: 'other'}


label2num = {'lying': 1,
 'sitting': 2,
 'standing': 3,
 'walking': 4,
 'running': 5,
 'cycling': 6,
 'Nordic walking': 7,
 'watching TV': 9,
 'computer work': 10,
 'car driving': 11,
 'ascending stairs': 12,
 'descending stairs': 13,
 'vacuum cleaning': 16,
 'ironing': 17,
 'folding laundry': 18,
 'house cleaning': 19,
 'playing soccer': 20,
 'rope jumping': 24,
 'other': 0}


data_columns = [ 
 'hand_acc_16g_x',
 'hand_acc_16g_y',
 'hand_acc_16g_z',
 'hand_gyroscope_x',
 'hand_gyroscope_y',
 'hand_gyroscope_z',
 'hand_magnometer_x',
 'hand_magnometer_y',
 'hand_magnometer_z',

 'chest_acc_16g_x',
 'chest_acc_16g_y',
 'chest_acc_16g_z',
 'chest_gyroscope_x',
 'chest_gyroscope_y',
 'chest_gyroscope_z',
 'chest_magnometer_x',
 'chest_magnometer_y',
 'chest_magnometer_z',


 'ankle_acc_16g_x',
 'ankle_acc_16g_y',
 'ankle_acc_16g_z',
 'ankle_gyroscope_x',
 'ankle_gyroscope_y',
 'ankle_gyroscope_z',
 'ankle_magnometer_x',
 'ankle_magnometer_y',
 'ankle_magnometer_z',
]

useful_columns =  data_columns  + ['activityID']
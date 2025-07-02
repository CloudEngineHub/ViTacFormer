from ha_data.data.naming_rule import *
from ha_data.data.train_meta import TrainMeta

def show_data(anno_j):
  anno_files = anno_j["url"]
  print('anno files:', len(anno_files), anno_files[0], anno_files[-1])
  data_key = anno_j["depend"]
  global meta
  data_j, data_version = meta.data.get(data_key.split('/')[-1])
  data_files = data_j["url"]
  print(data_version)
  print('data files:', len(data_files), data_files[0], data_files[-1])


# season_folder = '/mnt/netdata/Team/Robot/Data/train/double_hand_biglego_v0/1211'
# meta = TrainMeta(season_folder)
# anno_j, _ = meta.anno.get_from_simple_version('0.1.0'.split('.'))
# show_data(anno_j)

# anno_j, _ = meta.anno.get_from_simple_version('0.1.1'.split('.'))
# show_data(anno_j)

# anno_j, _ = meta.anno.get_from_simple_version('0.2.0'.split('.'))
# show_data(anno_j)


season_folder = '/mnt/netdata/Team/Robot/Data/train/double_hand_biglego_v0/1211'
meta = TrainMeta(season_folder)
anno_j, _ = meta.anno.get('0.2.2')
show_data(anno_j)

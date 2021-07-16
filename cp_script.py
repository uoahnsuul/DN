import os
import shutil
path_tuple = [('/home/dms1/media/TEST', '/home/dms1/media/dataset/TEST'),
              ('/home/dms1/sylim/TEST', '/home/dms1/media/dataset/TEST'),
              ('/home/dms1/media/VALIDATION', '/home/dms1/media/dataset/VALIDATION')]

# sample_path = '/home/dms1/media/TEST/UNFILTEREDRECON/RA_Tango2_3840x2160_60fps_10bit_37_RS0_POC9.npz'


if __name__=='__main__':

    def xgetFileList(dir, ext=('.bin')):
        matches = []
        for (path, dirnames, files) in os.walk(dir):
            for filename in files:
                if any(filename.endswith(e) for e in ext) and '/PREDICTION' not in path:
                    matches.append(os.path.join(path, filename))
        return matches

    for old_root_path, new_root_path in path_tuple:
        os.makedirs(new_root_path, exist_ok=True)
        file_list = [path for path in  xgetFileList(old_root_path, ext=('.npz', '.npy')) if '_RS0_' in path]
        total_len = len(file_list)
        for idx, file_path in enumerate(file_list):
            new_path = file_path.replace(old_root_path, new_root_path)
            if os.path.exists(new_path):
                continue
            if not os.path.exists(os.path.dirname(new_path)):
                os.makedirs(os.path.dirname(new_path))
            shutil.copy(file_path, new_path)
            print('[INFO] {} -> {} [{}/{}]'.format(file_path, new_path, idx+1, total_len))

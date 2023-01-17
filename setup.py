# encoding:utf-8
# ====================
# python 3.7.3
# 2020-09-20
# author: agt
# ====================

import os
import sys
from shutil import copyfile, rmtree
from distutils.core import setup
from Cython.Build import cythonize
import argparse
import yaml
import time

parser = argparse.ArgumentParser(description='Build files.')

parser.add_argument('--build_to', default="../end", type=str,
                    help='build files in which path, default ../end')
parser.add_argument('--build_type', default="pyd", type=str,
                    help='build so or type ')
parser.add_argument('--zipfile_name', default=None, type=str,
                    help='define network input size,default aiServer')


args = parser.parse_args()

# if sys.platform == 'win32':
#     print(f'暂不支持在WIN下编译此服务')
#     sys.exit()

print('+' * 32)
print('-' * 32)
print('- Start Compile Now')
print('- Welcome to the automation world')
print('- Copy Right By BMi')
print('-' * 32)
print('+' * 32)

# Step 0 环境准备
print('- Sync')
# os.popen('sync').read()
conf_file = 'config.yml'
with open(conf_file, 'r') as conf:
    config = conf.read()
config = yaml.load(config, Loader=yaml.FullLoader)

SOFT_VERSION = config['SystemConfig']['soft_version']
print(f'- version is {SOFT_VERSION}')
buildTime = time.strftime(time.strftime(
    '%Y%m%d%H%M%S', time.localtime(time.time())))
config['SystemConfig']['build_time'] = buildTime
with open(r'./_config.yml', 'w') as f:
    yaml.dump(config, f)
# 更新buildtime
os.remove(conf_file)
copyfile('_config.yml', conf_file)
os.remove('_config.yml')

if os.path.exists(args.build_to):
    print(f'--> remove {args.build_to}')
    rmtree(args.build_to)
os.makedirs(args.build_to)

# Step 1 编译
print('- Compile py to so')

# no need to compile
NoNeedCompile = [
    # 'app.py',
    'setup.py',
    # 'wsgi.py',
    'main.py',
]

# no need to move
NoNeedMove = [
    'clean.py',
    'setup.py',
    '__pycache__',
]

for root, dirs, files in os.walk(".", topdown=True):
    # print(f"root {root}, dirs {dirs}, files {files}")
    base_dir = os.path.join(args.build_to, root.replace('.', args.build_to))
    if not os.path.exists(base_dir) and '__pycache__' not in base_dir:
        print(f"--> mkdir base_dir {base_dir}")
        os.mkdir(base_dir)
    for d in dirs:
        if not os.path.exists(os.path.join(base_dir, d)) and '__pycache__' not in d:
            print(f"--> mkdir dir {d}")
            os.mkdir(os.path.join(base_dir, d))
    # 上面的代码是按照项目的目录再创建一个空的目录
    # sys.exit()
    for ff in files:
        # 只编译 .py 文件
        if ff[-3:] == '.py' and ff not in NoNeedCompile:
            print('-' * 32)
            print(f"-->> file {ff}")
            setup(ext_modules=cythonize([os.path.join(root, ff)], compiler_directives={'language_level': 3}),
                  script_args=["build_ext"])
            # 轮询编译后的临时文件目录，抓取{args.build_type}文件
            for ro, d, fi in os.walk('./build'):
                for f_name in fi:
                    if f_name[-1-len(args.build_type):] == '.'+args.build_type:
                        # 把{args.build_type}文件复制到它应该在的位置
                        f_list = f_name.split('.')
                        if len(f_list) > 2:
                            dist_f_name = f"{f_list[0]}.{args.build_type}"
                        else:
                            dist_f_name = f_name
                        print(f"---> {f_name}")
                        copyfile(os.path.join(ro, f_name),
                                 os.path.join(base_dir, dist_f_name))
                    # 删除临时文件
                    os.remove(os.path.join(ro, f_name))
            print('-' * 32)
        else:
            # 把非py文件直接放到对应的位置
            if '.'+args.build_type not in ff and ff not in NoNeedMove :
                copyfile(os.path.join(root, ff), os.path.join(base_dir, ff))

# 轮询删掉所有临时生成的.c文件
for root, dirs, files in os.walk(".", topdown=True):
    for ff in files:
        if ff[-2:] == '.c':
            os.remove(os.path.join(root, ff))

# 删除build临时目录

rmtree('./build')

print('- Compile so end.')

print('+' * 32)

# os.popen('sync').read()

# 加密压缩
if args.zipfile_name:
    cmd = f"zip -rP aiServer#HSyz {args.zipfile_name}.zip {args.build_to}"
    os.popen(cmd).read()

# os.popen('sync').read()
print('+' * 32)
print('-' * 32)
print('- End')
print('- Copy Right By BMi')
print('-' * 32)
print('+' * 32)

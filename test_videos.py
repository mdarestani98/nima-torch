import os


def main():
    root = 'video'
    model = 'ckpts_hkphi/epoch-41.pth'
    skip = 1
    save_root = 'video_pred'
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    save_suffix = '_pred'
    for video in os.listdir(root):
        if not video.endswith('.mp4'):
            continue
        print(f'Processing {video}')
        video_path = os.path.join(root, video)
        save_path = os.path.join(save_root, video[:-4] + save_suffix + '.mp4')
        os.system(f'python test_video.py --model {model} --video {video_path} --skip {skip} --save_video {save_path}')
        print(f'Predictions saved to {save_path}')


if __name__ == '__main__':
    main()

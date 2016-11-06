"""
Helper script to transform the Voxforge annotations into separate text files,
one for each example audio file.
"""

import os

voxforge_dir = './Voxforge'

speaker_folders = os.listdir(voxforge_dir)

for i, d in enumerate(speaker_folders):

    print('Processing {} / {} folders'.format(i, len(speaker_folders)))

    speaker_dir = os.path.join(voxforge_dir, d)

    # generate folder for txt files
    if not os.path.isdir(os.path.join(speaker_dir, 'txt')):
        os.makedirs(os.path.join(speaker_dir, 'txt'))

    # read prompts file
    with open(os.path.join(speaker_dir, 'etc', 'PROMPTS')) as f:
        content = f.readlines()

    for row in content:
        # get id
        id = row.split(' ')[0].split('/')[-1]

        # get prompt
        prompt = row.split(' ')[1:]
        prompt = ' '.join(prompt).replace('\n', '').lower()

        # write to file
        with open(os.path.join(speaker_dir, 'txt', id + '.txt'), 'w') as w:
            w.write(prompt)






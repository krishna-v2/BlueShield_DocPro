from pathlib import Path


class Cnfg:
    base_folder = Path(__file__).parent.parent
    temp_folder = base_folder.joinpath('temp')
    template_file1 = base_folder.joinpath('onboarding', 'BRC_Template1', 'brc_template.png')
    assert(base_folder.is_dir())
    assert(temp_folder.is_dir())
    assert(template_file1.is_file())

    address_pos1 = {'hs': 190, 'he': 371, 'ws': 432, 'we': 1288}
    barcode_pos1 = {'hs': 351, 'he': 421, 'ws': 423, 'we': 1270}
    email_pos1 = {'hs': 518, 'he': 589, 'ws': 826, 'we': 1700}
    phone_pos1 = {'hs': 470, 'he': 520, 'ws': 620, 'we': 1395}
    s = 27
    checkbox_pos1 = {'hs': 502 - s * 3, 'he': 502 + s * 3, 'ws': 488 - s, 'we': 488 + s}
    canvas_shape = (606, 904)


if __name__ == '__main__':
    cnfg = Cnfg()


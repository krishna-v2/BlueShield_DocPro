from pathlib import Path
from os.path import join

class Cnfg:
    base_folder = Path(__file__).parent.parent
    temp_folder = base_folder.joinpath('temp')
    template_file1 = base_folder.joinpath('onboarding', 'BRC_Template1', 'brc_template.png')
    assert(base_folder.is_dir())
    assert(temp_folder.is_dir())
    assert(template_file1.is_file())

    canvas_h_offset = 100
    address_pos1 = {'hs': 190, 'he': 371, 'ws': 432, 'we': 1288, 'canvas_h_offset': 0, 'canvas_w_offset': 0}
    phone_offset = address_pos1['canvas_h_offset'] + address_pos1['he'] - address_pos1['hs'] + canvas_h_offset
    phone_pos1 = {'hs': 470, 'he': 520, 'ws': 620, 'we': 1395, 'canvas_h_offset': phone_offset, 'canvas_w_offset': 0}
    left_parentheses1 = {'hs': phone_pos1['hs'], 'he': phone_pos1['he'], 'ws': 621, 'we': 642,
                         'canvas_h_offset': phone_offset, 'canvas_w_offset': 0}
    right_parentheses1 = {'hs': phone_pos1['hs'], 'he': phone_pos1['he'], 'ws': 818, 'we': 839,
                         'canvas_h_offset': phone_offset, 'canvas_w_offset': 197}

    email_offset = phone_pos1['canvas_h_offset'] + phone_pos1['he'] - phone_pos1['hs'] + canvas_h_offset
    email_pos1 = {'hs': 518, 'he': 589, 'ws': 826, 'we': 1640, 'canvas_h_offset': email_offset, 'canvas_w_offset': 0}
    barcode_pos1 = {'hs': 351, 'he': 421, 'ws': 423, 'we': 1270}

    s = 27
    checkbox_pos1 = {'hs': 502 - s * 3, 'he': 502 + s * 3, 'ws': 488 - s, 'we': 488 + s}
    canvas_shape = (606, 904)
    pos = {'address': address_pos1, 'phone': phone_pos1, 'email': email_pos1, 'barcode': barcode_pos1,
           'checkbox': checkbox_pos1, 'left_p': left_parentheses1, 'right_p': right_parentheses1}
    text_kinds = ['address', 'phone', 'email']
    all_kinds = ['address', 'phone', 'email', 'checkbox', 'barcode']

class EmailP:
    cnfg = Cnfg()
    pcts = list(range(200, 400, 25))
    save_folder = join(cnfg.temp_folder, 'email')

if __name__ == '__main__':
    cnfg = Cnfg()
    print(cnfg)


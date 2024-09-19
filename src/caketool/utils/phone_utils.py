import re

def norm_vn_phone(phone: str):
    if not phone:
        return None
    phone = phone.replace(" ", "")

    # Regular expressions for validating and cleaning up phone numbers
    reg_phone_list = [
        re.compile(r'((3([2-9]))|(5([25689]))|(7([0|6-9]))|(8([1-9]))|(9([0-9])))([0-9]{7})'),
        re.compile(r'(\+?84|0)((3([2-9]))|(5([25689]))|(7([0|6-9]))|(8([1-9]))|(9([0-9])))([0-9]{7})'),
        re.compile(r'((16([2-9]))|(12([0-9]))|(18(6|9))|(199))([0-9]{7})'),
        re.compile(r'(\+?84|0)((16([2-9]))|(12([0-9]))|(18(6|9))|(199))([0-9]{7})'),
        re.compile(r'(\+)\d+'),
        re.compile(r'(2([1-9])([0-9]){8})'),
        re.compile(r'(\+?84|0)(2([1-9])([0-9]){8})')
    ]

    clean_data = {
        '+843': re.compile(r'(\+?84|0)16|^16'),

        '+8470': re.compile(r'(\+?84|0)120|^120'),
        '+8479': re.compile(r'(\+?84|0)121|^121'),
        '+8477': re.compile(r'(\+?84|0)122|^122'),
        '+8476': re.compile(r'(\+?84|0)126|^126'),
        '+8478': re.compile(r'(\+?84|0)128|^128'),

        '+8483': re.compile(r'(\+?84|0)123|^123'),
        '+8484': re.compile(r'(\+?84|0)124|^124'),
        '+8485': re.compile(r'(\+?84|0)125|^125'),
        '+8481': re.compile(r'(\+?84|0)127|^127'),
        '+8482': re.compile(r'(\+?84|0)129|^129'),

        '+8456': re.compile(r'(\+?84|0)186|^186'),
        '+8458': re.compile(r'(\+?84|0)188|^188'),
        '+8459': re.compile(r'(\+?84|0)199|^199')
    }

    # Validating Vietnamese phone number
    is_valid_phone = re.compile(r'^(\+?84|0)((((3([2-9]))|(5([25689]))|(7([0|6-9]))|(8([1-9]))|(9([0-9])))([0-9]{7}))|(2([1-9])([0-9]){8}))$')

    # Find the matching format for the phone number
    vn_phone_type = next((i for i, reg in enumerate(reg_phone_list) if reg.match(phone)), -1)

    # Normalize phone based on the matching format
    if vn_phone_type in [0, 5]:
        phone = '+84' + phone
        return phone if is_valid_phone.match(phone) else None
    elif vn_phone_type in [1, 6]:
        phone = re.sub(r'^(84|0)', '+84', phone)
        return phone if is_valid_phone.match(phone) else None
    elif vn_phone_type in [2, 3]:
        for k, v in clean_data.items():
            phone = v.sub(k, phone)
        return phone if is_valid_phone.match(phone) else None
    elif vn_phone_type == 4:
        return phone
    else:
        return None
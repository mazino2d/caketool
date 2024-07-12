# Define a translation table to convert Vietnamese characters to their ASCII equivalents
VN_EN_TRANS = str.maketrans(
    "ÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬĐÈÉẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴ"
    "áàảãạăắằẳẵặâấầẩẫậđèéẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ",
    "A"*17 + "D" + "E"*11 + "I"*5 + "O"*17 + "U"*11 + "Y"*5 +
    "a"*17 + "d" + "e"*11 + "i"*5 + "o"*17 + "u"*11 + "y"*5,
    chr(774) + chr(770) + chr(795) + chr(769) + chr(768) + chr(777) + chr(771) + chr(803)
)

UPPER_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
LOWER_ALPHABET = "abcdefghijklmnopqrstuvwxyz"


def remove_vn_diacritics(txt: str) -> str:
    """
    Converts Vietnamese characters with diacritics to their corresponding
    ASCII equivalents.

    Args:
        txt (str): The Vietnamese text to be converted.

    Returns:
        str: The converted text with diacritics removed.
    """
    return txt.translate(VN_EN_TRANS)

# Define a translation table to convert Vietnamese characters to their ASCII equivalents
VN_EN_TRANS = str.maketrans(
    "ÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬĐÈÉẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴ"
    "áàảãạăắằẳẵặâấầẩẫậđèéẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ",
    "A" * 17
    + "D"
    + "E" * 11
    + "I" * 5
    + "O" * 17
    + "U" * 11
    + "Y" * 5
    + "a" * 17
    + "d"
    + "e" * 11
    + "i" * 5
    + "o" * 17
    + "u" * 11
    + "y" * 5,
    chr(774) + chr(770) + chr(795) + chr(769) + chr(768) + chr(777) + chr(771) + chr(803),
)

UPPER_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
LOWER_ALPHABET = "abcdefghijklmnopqrstuvwxyz"


def remove_vn_diacritics(txt: str) -> str:
    """Convert Vietnamese text with diacritics to plain ASCII equivalents.

    Replaces all accented Vietnamese characters (e.g. ``ă``, ``ơ``, ``ệ``)
    with their closest unaccented ASCII counterparts using a pre-built
    translation table.  The special character ``đ``/``Đ`` is mapped to
    ``d``/``D``.

    Parameters
    ----------
    txt : str
        Input text that may contain Vietnamese diacritical characters.

    Returns
    -------
    str
        Text with all diacritics removed, preserving original case for
        non-accented characters.

    Examples
    --------
    >>> remove_vn_diacritics("Nguyễn Văn Bình")
    'Nguyen Van Binh'
    >>> remove_vn_diacritics("Hà Nội")
    'Ha Noi'
    """
    return txt.translate(VN_EN_TRANS)

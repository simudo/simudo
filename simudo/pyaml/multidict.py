
from generic_escape import GenericEscape

class EscapeMultidictKey(GenericEscape):
    escaped = {"\\": "\\\\",
               "@": r"\\@"}
EscapeMultidictKey_instance = EscapeMultidictKey()

def multidict_items(mapping):
    for key, value in mapping.items():
        endp, strings = EscapeMultidictKey_instance.unescape_split(key, '@', maxsplit=1)
        realkey = strings[0]
        yield (realkey, value)


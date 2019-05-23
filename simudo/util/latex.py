
__all__ = ['sci_to_latex1']

def sci_to_latex1(string, suppress_mantissa_eq_1=True):
    '''Convert a number in scientific notation to reasonable LaTeX code.

Do NOT modify this method! Copy-paste it and change the name. Other
code may rely on its exact behavior (bug-for-bug compatibility).
'''
    mantissa, expt = string.lower().split('e')
    if mantissa[0] in '+-':
        sign = mantissa[0]
        mantissa = mantissa[1:]
    else:
        sign = ''

    if suppress_mantissa_eq_1 and mantissa == '1':
        mantissa = None

    if mantissa is not None:
        mantissa = mantissa + r'\times'
    else:
        mantissa = ''

    return (r"{}{}10^{{{}}}".format(sign, mantissa, expt))


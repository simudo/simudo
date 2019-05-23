
import logging

def is_under(prefix, x):
    return x == prefix or x.startswith(prefix + '.')

class NameLevelFilter(logging.Filter):
    def __init__(self, name_levelno_rules, *args, **kwargs):
        self.name_levelno_rules = name_levelno_rules
        super().__init__(*args, **kwargs)

    def filter(self, record):
        name, level = record.name, record.levelno
        for rule_name, rule_level in self.name_levelno_rules:
            if is_under(rule_name, name) and level >= rule_level:
                return True
        return False

def main():
    logging.getLogger('').setLevel(logging.NOTSET)

    filename = 'debug.log'
    fileha = logging.FileHandler(
        filename=filename, mode='w')
    fileha.setFormatter(
        logging.Formatter(
            '%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'))

    console = logging.StreamHandler()
    console.setFormatter(
        logging.Formatter(
            '%(asctime)s %(name)-12s: %(levelname)-8s %(message)s',
            datefmt='%H:%M:%S'))

    logging.getLogger('').addHandler(fileha)
    logging.getLogger('').addHandler(console)

    fileha.addFilter(NameLevelFilter([
        ('root', logging.DEBUG),
        ('A', logging.ERROR),
        ('B', logging.DEBUG)]))
    console.addFilter(NameLevelFilter([
        ('root', logging.ERROR),
        ('A', logging.DEBUG),
        ('B', logging.ERROR)]))

    for k in ['A.X', 'A', 'B', '']:
        log = logging.getLogger(k)
        log.debug("debug@"+k)
        log.error("error@"+k)

if __name__ == '__main__':
    main()


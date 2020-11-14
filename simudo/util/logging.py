# copyright 2019 Eduard Christian Dumitrescu
# license: CC0 / https://creativecommons.org/publicdomain/zero/1.0/

import logging
import os

from cached_property import cached_property

from .setattr_init_mixin import SetattrInitMixin

__all__ = [
    'NameLevelFilter',
    'TypicalLoggingSetup']

def is_under(prefix, x):
    return x == prefix or prefix == '' or x.startswith(prefix + '.')

class NameLevelFilter(logging.Filter):
    def __init__(self, name_levelno_rules, *args, **kwargs):
        self.name_levelno_rules = name_levelno_rules
        super().__init__(*args, **kwargs)

    def filter(self, record):
        name, level = record.name, record.levelno
        for rule_name, rule_level in self.name_levelno_rules:
            if is_under(rule_name, name):
                return level >= rule_level
        return False

class TypicalLoggingSetup(SetattrInitMixin):
    """Class that sets up logging and filtering in a typical way for Simudo.

Parameters
----------
dolfin: bool, optional
    Configure the dolfin log level as well. Note that this imports
    ``dolfin``, which takes a while. Only use it if you're okay with
    that. (default: True)
truncate: bool, optional
    Truncate (delete) the log file contents before starting to write
    to it. (default: False)
"""
    dolfin = True
    truncate = True

    @property
    def _mode(self):
        return "w" if self.truncate else "a"

    def ensure_parent_dir(self, filename):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError:
            pass

    @cached_property
    def logfile_formatter(self):
        return logging.Formatter(
            '%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S')

    @cached_property
    def console_formatter(self):
        return logging.Formatter(
            '%(asctime)s %(name)-12s: %(levelname)-8s %(message)s',
            datefmt='%H:%M:%S')

    @property
    def debug_filename(self):
        return self.filename_prefix + 'debug.log'

    @property
    def info_filename(self):
        return self.filename_prefix + 'info.log'

    @cached_property
    def stream_debug(self):
        self.ensure_parent_dir(self.debug_filename)
        h = logging.FileHandler(filename=self.debug_filename, mode=self._mode)
        h.setFormatter(self.logfile_formatter)
        return h

    @cached_property
    def stream_info(self):
        self.ensure_parent_dir(self.info_filename)
        h = logging.FileHandler(filename=self.info_filename, mode=self._mode)
        h.setFormatter(self.logfile_formatter)
        return h

    @cached_property
    def stream_console(self):
        h = logging.StreamHandler()
        h.setFormatter(self.console_formatter)
        return h

    def setup_handlers(self):
        for h in (self.stream_debug,
                  self.stream_info,
                  self.stream_console):
            logging.getLogger('').addHandler(h)

    def setup_filters(self):
        self.stream_debug.addFilter(NameLevelFilter([
            ('FFC', logging.INFO),
            ('UFL', logging.INFO),
            # ('assign', logging.DEBUG),
            ('matplotlib', logging.INFO),
            ('', logging.INFO)]))
        self.stream_info.addFilter(NameLevelFilter([
            ('FFC', logging.ERROR),
            ('UFL', logging.ERROR),
            ('', logging.INFO)]))
        self.stream_console.addFilter(NameLevelFilter([
            ('FFC', logging.ERROR),
            ('UFL', logging.ERROR),
            ('', logging.INFO)]))

    def setup_logging(self):
        logging.getLogger('').setLevel(logging.NOTSET)

    def setup_dolfin_loglevel(self):
        if self.dolfin:
            import dolfin
            dolfin.set_log_level(50)

    def setup(self):
        self.setup_logging()
        self.setup_handlers()
        self.setup_filters()
        self.setup_dolfin_loglevel()

__author__ = 'matsrichter'

class Error_module:

    def __index__(self,module_list):
        self.module_list()
        self.error = False
        return

    def get_function(self,filepointer,metapointer = None ):
        """
        This functions looks for error-flags in all modules, if modules are flagged with self.error == True, the module
        will return 1.0 else 0.
        The function also resets all error flags to False after they were checked.

        :param filepointer: filepointer to a pdf
        :param metapointer: metadata of the file
        :return: float64, in this case 0 (no Error) and 1 (Error)
        """
        error = 0
        for m in self.module_list:
            if m.error:
                m.error = False
                error = 1.0
        return error
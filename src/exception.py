import sys


def error_message_detail(error, error_detail:sys)->str:
    
    '''
    return custom designed error message.

    About exc_info():
        sys.exc_info() function returns a 3- tuple with the exception, 
        the exception's parameter, and a traceback object that pinpoints 
        the line of Python that raised the exception.
    '''

    _, _, exc_traceback = error_detail.exc_info()
    file_name = exc_traceback.tb_frame.f_code.co_filename # File in which exception has occurred.
    
    # Custom-designed error message.
    error_message = "Error occurred in python script name [{0}] line number [{1}] error message [{2}]".format(file_name, exc_traceback.tb_lineno, str(error))

    return error_message

class CustomException(Exception):
    
    def __init__(self, error_message, error_detail:sys):    
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)
    
    def __str__(self) -> str:
        return self.error_message

    
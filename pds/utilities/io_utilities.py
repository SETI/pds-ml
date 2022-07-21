# This module contains basic input/output utilities

#******************************************************************************
# Print dictionary name-value pairs to the standard output.
#
# This function can be used to quickly view the attributes of an object.
#
# It is very useful to use in the __repr__ or __str__ to view the attributes of an object.
#    
# INPUTS
#   dictionary  : [dict or object] A dictionary object of <attribute, value> pairs specifying the 
#                configuration parameters. If an object is passed then the object is first converted to a dict
#   verbosity   : [bool] If True then print the dictionary. Otherwise, just return as a string
#
# OUTPUTS
#   Prints a summary to the standard output, if request
#   dict_string : [str] contains the formated dictionary attributes    
#******************************************************************************
def print_dictionary(dictionary, verbosity=False):

    if (not isinstance(dictionary, dict) and isinstance(dictionary, object)):
        dictionary = dictionary.__dict__

    if not isinstance(dictionary, dict):
        raise Exception ('Passed dictionary does not appear to be of type dict')

    dict_string = '\n'
    for key in dictionary.keys():
        dict_string += '\t{} = {}\n'.format(key, dictionary[key])

    if verbosity:
        print(dict_string)
 
    return dict_string



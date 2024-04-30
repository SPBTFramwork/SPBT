from .transform_tokensub import *
from .transform_invichar import *
from .transform_deadcode import *
from .transform_clean import *
from .transform_identifier_name import *
from .transform_bracket import *
from .transform_augmented_assignment import *
from .transform_cmp import *
from .transform_for_update import *
from .transform_array_definition import *
from .transform_array_access import *
from .transform_declare_lines import *
from .transform_declare_position import *
from .transform_declare_assign import *
from .transform_for_format import *
from .transform_for_while import *
from .transform_loop_infinite import *
from .transform_break_goto import *
from .transform_if_exclamation import *
from .transform_if_return import *
from .transform_if_switch import *
from .transform_if_nested import *
from .transform_if_else import *

transformation_operators = {
    'tokensub': {
        'sh':   (match_tokensub_identifier,
                convert_tokensub_sh,
                count_tokensub)
    },

    'invichar': {
        'ZWSP': (match_invichar_identifier,
                 convert_invichar_ZWSP,
                 count_invichar),

        'ZWNJ': (match_invichar_identifier,
                 convert_invichar_ZWNJ,
                 count_invichar),
        
        'LRO': (match_invichar_identifier,
                 convert_invichar_LRO,
                 count_invichar),

        'BKSP': (match_invichar_identifier,
                 convert_invichar_LRO,
                 count_invichar)
    },

    'deadcode':{
        'deadcode1': (match_function, 
                     convert_deadcode1,
                     count_deadcode),
        'deadcode2': (match_function, 
                     convert_deadcode2,
                     count_deadcode),
    },

    'clean':{
        'clean': (match_nothing, convert_nothing, count_nothing)
    },

    'identifier_name': {

        'camel':            (match_identifier, 
                             convert_camel,           
                             count_camel),

        'pascal':           (match_identifier, 
                             convert_pascal,          
                             count_pascal),

        'snake':            (match_identifier, 
                             convert_snake,           
                             count_snake),

        'hungarian':        (match_identifier, 
                             convert_hungarian,       
                             count_hungarian),

        'init_underscore':  (match_identifier, 
                             convert_init_underscore, 
                             count_init_underscore),

        'init_dollar':      (match_identifier, 
                             convert_init_dollar,     
                             count_init_dollar),

        'upper':            (match_identifier, 
                             convert_upper,           
                             count_upper),

        'lower':            (match_identifier, 
                             convert_lower,           
                             count_lower),
    },

    'bracket': {

        'del_bracket':      (match_ifforwhile_has_bracket,    
                             convert_del_ifforwhile_bracket,  
                             count_hasnt_ifforwhile_bracket),

        'add_bracket':      (match_ifforwhile_hasnt_bracket, 
                             convert_add_ifforwhile_bracket, 
                             count_has_ifforwhile_bracket),
    },
    
    'augmented_assignment':{
        'non_augmented':    (match_augmented_assignment, 
                             convert_non_augmented_assignment,
                             count_non_augmented_assignment),

        'augmented':         (match_non_augmented_assignment, 
                              convert_augmented_assignment,
                              count_augmented_assignment),
    },

    'cmp': {
        'smaller':      (match_cmp, 
                         convert_smaller,    
                         count_smaller),

        'bigger':       (match_cmp, 
                         convert_bigger,     
                         count_bigger),

        'equal':        (match_cmp, 
                         convert_equal,      
                         count_equal),

        'not_equal':    (match_cmp, 
                         convert_not_equal,  
                         count_not_equal),
    },

    'for_update': {
        'left':         (match_not_left,         
                         convert_left,        
                         count_left),

        'right':        (match_not_right,        
                         convert_right,
                         count_right),

        'augment':      (match_not_augment,      
                         convert_augment,     
                         count_augment),

        'assignment':   (match_not_assignment,   
                         convert_assignment,  
                         count_assignment)
    },

    'array_definition':{
        'dyn_mem':      (match_static_mem,   
                         convert_dyn_mem,      
                         count_dyn_mem),

        'static_mem':   (match_dyn_mem,         
                         convert_static_mem,   
                         count_static_mem),
    },

    'array_access': {
        'pointer':  (match_array,    convert_pointer,   count_pointer),
        'array':    (match_pointer,    convert_array,     count_array),
    },
    
    'declare_lines': {
        'split': (match_lines_merge,      
                  convert_lines_split,      
                  count_lines_split),

        'merge': (match_lines_split,      
                  convert_lines_merge,      
                  count_lines_merge),
    },
    
    'declare_position': {
        'first': (match_not_first,  
                  convert_first,
                  count_first),

        'temp':  (match_not_tmp,    
                  convert_temp,
                  count_temp),
    },

    'declare_assign': {
        'split': (match_assign_merge, 
                  convert_assign_split, 
                  count_assign_split),

        'merge': (match_assign_split, 
                  convert_assign_merge, 
                  count_assign_merge),
    },

    'for_format': {
        'abc':      (match_for,     convert_abc, count_abc),
        'obc':      (match_for,     convert_obc, count_obc),
        'aoc':      (match_for,     convert_aoc, count_aoc),
        'abo':      (match_for,     convert_abo, count_abo),
        'aoo':      (match_for,     convert_aoo, count_aoo),
        'obo':      (match_for,     convert_obo, count_obo),
        'ooc':      (match_for,     convert_ooc, count_ooc),
        'ooo':      (match_for,     convert_ooo, count_ooo),
    },

    'for_while': {
        'for':      (match_loop,   
                     convert_for,
                     count_for),

        'while':    (match_loop,     
                     convert_while,
                     count_while),

        'do_while': (match_loop,     
                     convert_do_while,
                     count_do_while),
    },

    'loop_infinite': {
        # 'finite_for': (), 
        # 'infinite_for': (),
        # 'finite_while': (),
        'infinite_while': (match_for_while, cvt_infinite_while, count_inf_while),
    },

    'break_goto': {
        'goto': (match_break, 
                 cvt_break2goto),

        'break': ()
    },

    'if_exclamation':{
        'not_exclamation': (),

        'exclamation': (match_if_equivalent, 
                        cvt_equivalent, match_if_equivalent),
    },

    'if_return':{
        'not_return': (),

        'return': (match_if_return, 
                   cvt_return),
    },

    'if_switch':{
        'switch':   (match_if,
                    cvt_if2switch,
                    count_switch),

        'if':       (match_switch, 
                    cvt_switch2if,
                    count_if),
    },

    'if_nested':{
        'not_nested': (),
        'nested': (match_if_split, cvt_split, count_if_split),  
    },

    'if_else': {
        'not_else': (),
        'else':     (match_if_add_else, cvt_add_else),
    }
}
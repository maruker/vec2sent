from vec2sent.sys_path_hack import add_to_path

add_to_path(["quick_thought", "S2V", "src"])

from .S2V.src import encoder_manager, configuration
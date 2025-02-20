create_clock -period [expr [getenv ECE411_CLOCK_PERIOD_PS] / 1000.0] -name my_clk clk
set_fix_hold [get_clocks my_clk]

# set_input_delay  0.2 [get_ports lookup_pc]           -clock my_clk
# set_input_delay  0.2 [get_ports lookup_valid]        -clock my_clk
# set_output_delay 0.2 [get_ports lookup_prediction]   -clock my_clk
# set_output_delay 0.2 [get_ports lookup_target]       -clock my_clk
# set_output_delay 0.2 [get_ports lookup_info]         -clock my_clk
# set_output_delay 0.2 [get_ports lookup_ready]        -clock my_clk

# set_input_delay  0.2 [get_ports update_pc]           -clock my_clk
# set_input_delay  0.2 [get_ports update_prediction]   -clock my_clk
# set_input_delay  0.2 [get_ports update_info]         -clock my_clk
# set_input_delay  0.2 [get_ports update_actual]       -clock my_clk
# set_input_delay  0.2 [get_ports update_opcode]       -clock my_clk
# set_input_delay  0.2 [get_ports update_target]       -clock my_clk
# set_input_delay  0.2 [get_ports update_valid]        -clock my_clk

set_load 0.1 [all_outputs]
set_max_fanout 1 [all_inputs]
set_fanout_load 8 [all_outputs]

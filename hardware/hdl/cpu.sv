module cpu
import types::*;
(
  input logic            clk,
  input logic            rst,
  input types::pixels_t pixels,
  output logic [1:0] [511:0] lines,
  output logic [1:0] flag
);

  types::header_residual_reg hr_reg, hr_reg_next;
  types::residual_compress_reg cr_reg, cr_reg_next;
  types::compress_commit_reg cc_reg, cc_reg_next;

  header header_inst(.clk(clk), .rst(rst), .pixels(pixels), .hr_reg(hr_reg_next));
  residual residual_inst(.clk(clk), .rst(rst), .hr_reg(hr_reg), .cr_reg(cr_reg_next));
  compress compress_inst(.clk(clk), .rst(rst), .cr_reg(cr_reg), .cc_reg(cc_reg_next));

  always @(posedge clk) begin
    hr_reg <= hr_reg_next;
    cr_reg <= cr_reg_next;
    cc_reg <= cc_reg_next;
  end

  always_comb begin
    lines[0] = cc_reg.lines.l1.raw;
    lines[1] = cc_reg.lines.l2.raw;
    flag = cc_reg.flag;
  end


endmodule

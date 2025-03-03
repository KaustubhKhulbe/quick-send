module cpu
import types::*;
(
  input logic            clk,
  input logic            rst,
  input types::pixels_t pixels,
  output types::header_residual_reg hr_reg,
  output types::residual_compress_reg cr_reg
);

  types::header_residual_reg hr_reg_next;
  types::residual_compress_reg cr_reg_next;

  header header_inst(.clk(clk), .rst(rst), .pixels(pixels), .hr_reg(hr_reg_next));
  residual residual_inst(.clk(clk), .rst(rst), .hr_reg(hr_reg), .cr_reg(cr_reg_next));

  always @(posedge clk) begin
    hr_reg <= hr_reg_next;
    cr_reg <= cr_reg_next;
  end

endmodule

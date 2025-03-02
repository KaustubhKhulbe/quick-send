module cpu
import types::*;
(
  input logic            clk,
  input logic            rst,
  input types::pixels_t pixels,
  output types::header_residual_reg hr_reg
);

  types::header_residual_reg res, res_next;

  header header_inst(.clk(clk), .rst(rst), .pixels(pixels), .hr_reg(res_next));

  always @(posedge clk) begin
    res <= res_next;
  end

  always_comb begin
    hr_reg = res;
    end

endmodule

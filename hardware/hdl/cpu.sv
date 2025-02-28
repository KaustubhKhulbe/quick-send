module cpu
import types::*;
(
  input logic clk,
  input logic rst,
  input logic [3:0][7:0] pixels [31:0],
  output logic compressable,
  output logic [47:0] h
);

  types::residual_reg res, res_next;

  header header_inst(.clk(clk), .pixels(pixels), .compressable(res.compressable), .h(res.header));

  always @(posedge clk) begin
    res_next <= res;
  end

  always_comb begin
    compressable = res_next.compressable;
    h = res_next.header;
    end
endmodule

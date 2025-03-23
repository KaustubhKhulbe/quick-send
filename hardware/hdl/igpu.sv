module igpu
import types::*;
(
  input logic            clk,
  input logic            rst,
  input types::pixels_t pixels,
  output logic [511:0] lines,
  output logic [1:0] flag
);

  types::header_residual_reg hr_reg, hr_reg_next;
  types::residual_compress_reg cr_reg, cr_reg_next;
  types::compress_commit_reg cc_reg, cc_reg_next;

  header header_inst(.pixels(pixels), .hr_reg(hr_reg_next));
  residual residual_inst(.hr_reg(hr_reg), .cr_reg(cr_reg_next));
  compress compress_inst(.cr_reg(cr_reg), .cc_reg(cc_reg_next));

  always @(posedge clk) begin
    if (rst) begin
      hr_reg <= types::header_residual_reg'(1105'b0);
      cr_reg <= types::residual_compress_reg'(1073'b0);
      cc_reg <= types::compress_commit_reg'(515'b0);
    end
    else begin
      hr_reg <= hr_reg_next;
      cr_reg <= cr_reg_next;
      cc_reg <= cc_reg_next;
    end
  end

  always_comb begin
    lines = cc_reg.lines.l1.raw;
    flag = cc_reg.flag;
  end


endmodule

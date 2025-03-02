module top_tb;

  timeunit 1ms;
  timeprecision 1ps;
  import types::*;

  bit clk;
  bit rst;
  types::pixels_t pixels;
  types::header_residual_reg hr_reg;

  logic [7:0] r_min, g_min, b_min, a_min;

  cpu dut(.clk(clk), .rst(rst), .pixels(pixels), .hr_reg(hr_reg));

  initial begin
    rst = 1'b1;
    # 20;
    rst = 1'b0;
    # 20;

    for (int c = 0; c < 100; c++) begin
      for (int i = 0; i < 32; i = i + 1) begin
      for (int j = 0; j < 4; j = j + 1) begin
          pixels.pixels[i][j] = byte'($urandom % 256);;
          end
      end

      # 5;

      r_min = 8'hFF;
      g_min = 8'hFF;
      b_min = 8'hFF;
      a_min = 8'hFF;

      for (int i = 0; i < 32; i = i + 1) begin
      r_min = (pixels.pixels[i][0] < r_min) ? pixels.pixels[i][0] : r_min;
      g_min = (pixels.pixels[i][1] < g_min) ? pixels.pixels[i][1] : g_min;
      b_min = (pixels.pixels[i][2] < b_min) ? pixels.pixels[i][2] : b_min;
      a_min = (pixels.pixels[i][3] < a_min) ? pixels.pixels[i][3] : a_min;
      end

      assert(hr_reg.header.min_values.r_min == r_min);
      assert(hr_reg.header.min_values.g_min == g_min);
      assert(hr_reg.header.min_values.b_min == b_min);
      assert(hr_reg.header.min_values.a_min == a_min);
    end

    $display("All tests passed!");
    $finish;
    end


  initial begin
    clk = 1'b0;
    forever #1 clk = ~clk;
    end

endmodule

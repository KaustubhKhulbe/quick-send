module top_tb_archived;

  timeunit 1ms;
  timeprecision 1ps;
  import types::*;

  bit clk;
  bit rst;
  types::pixels_t pixels;
  logic [1:0] [511:0] lines;
  logic [1:0] flag;

  logic [7:0] r_min, g_min, b_min, a_min;
  logic [7:0] r_max, g_max, b_max, a_max;

  igpu dut(.clk(clk), .rst(rst), .pixels(pixels), .lines(lines), .flag(flag));

  initial begin
    rst = 1'b1;
    # 20;
    rst = 1'b0;
    # 20;

    for (int c = 0; c < 100; c++) begin
      for (int i = 0; i < 32; i = i + 1) begin
      for (int j = 0; j < 4; j = j + 1) begin
          pixels.pixels[i][j] = byte'($urandom % 7);
          end
      end

      # 5;

      r_min = 8'hFF;
      g_min = 8'hFF;
      b_min = 8'hFF;
      a_min = 8'hFF;

      r_max = 8'h00;
      g_max = 8'h00;
      b_max = 8'h00;
      a_max = 8'h00;

      for (int i = 0; i < 32; i = i + 1) begin
      r_min = (pixels.pixels[i][0] < r_min) ? pixels.pixels[i][0] : r_min;
      g_min = (pixels.pixels[i][1] < g_min) ? pixels.pixels[i][1] : g_min;
      b_min = (pixels.pixels[i][2] < b_min) ? pixels.pixels[i][2] : b_min;
      a_min = (pixels.pixels[i][3] < a_min) ? pixels.pixels[i][3] : a_min;

      r_max = (pixels.pixels[i][0] > r_max) ? pixels.pixels[i][0] : r_max;
      g_max = (pixels.pixels[i][1] > g_max) ? pixels.pixels[i][1] : g_max;
      b_max = (pixels.pixels[i][2] > b_max) ? pixels.pixels[i][2] : b_max;
      a_max = (pixels.pixels[i][3] > a_max) ? pixels.pixels[i][3] : a_max;
      end

      // assert(hr_reg.header.min_values.r_min == r_min);
      // assert(hr_reg.header.min_values.g_min == g_min);
      // assert(hr_reg.header.min_values.b_min == b_min);
      // assert(hr_reg.header.min_values.a_min == a_min);

      // assert(hr_reg.max_pixels.r_max == r_max);
      // assert(hr_reg.max_pixels.g_max == g_max);
      // assert(hr_reg.max_pixels.b_max == b_max);
      // assert(hr_reg.max_pixels.a_max == a_max);

      # 2;

      $display(dut.residual_inst.cr_reg.compressable);
      $display(flag);
      $display(lines);

    end

    $display("All tests passed!");
    $finish;
    end


  initial begin
    clk = 1'b0;
    forever #1 clk = ~clk;
    end

endmodule

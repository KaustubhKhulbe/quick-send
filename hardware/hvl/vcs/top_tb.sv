module top_tb;

  timeunit 1ms;
  timeprecision 1ps;
  import types::*;

  bit clk;
  bit rst;
  types::pixels_t pixels;
  logic [511:0] lines;
  logic [1:0] flag;
  logic [1:0] [511:0] expected_lines;
  logic [1:0] expected_flag;
  int res;

  igpu dut(.clk(clk), .rst(rst), .pixels(pixels), .lines(lines), .flag(flag));

  // Read pixels from a file
  task read_pixels_from_file(string filename);
    integer file, i;
    string line;
    integer r, g, b, a;

    file = $fopen(filename, "r");
    if (file == 0) begin
      $fatal("ERROR: Could not open file %s", filename);
    end

    i = 0;
    while (!$feof(file) && i < 32) begin
      line = "";
      void'($fgets(line, file));  // Read a line from the file
      if ($sscanf(line, "%d %d %d %d", r, g, b, a) == 4) begin
        pixels.channels.r_channel[i] = r[7:0];
        pixels.channels.g_channel[i] = g[7:0];
        pixels.channels.b_channel[i] = b[7:0];
        pixels.channels.a_channel[i] = a[7:0];
      end
      i++;
    end

    $fclose(file);
  endtask

  // task to read expected lines and flag from a file
  task read_expected_output(string filename);
  int fd;
  string line_data;
  
  fd = $fopen(filename, "r");
  if (fd == 0) begin
    $display("Error: Cannot open %s", filename);
    $finish;
  end

  // read two 512-bit binary values for lines[0] and lines[1]
  for (int i = 0; i < 2; i++) begin
    res = $fgets(line_data, fd);
    expected_lines[i] = '0;

    // convert each character ('0' or '1') into a 512-bit logic vector
    for (int j = 0; j < 512; j++) begin
      expected_lines[i][511 - j] = (line_data[j] == "1") ? 1'b1 : 1'b0;
    end
  end

  // read flag values
  res = $fscanf(fd, "%d %d\n", expected_flag[1], expected_flag[0]);

  $fclose(fd);
endtask

  // task to compare output
  task compare_output();
    // $display("===================================");
    // $display(dut.residual_inst.header.min_values.bits_required[11:9]);
    // $display(dut.residual_inst.header.min_values.bits_required[8:6]);
    // $display(dut.residual_inst.header.min_values.bits_required[5:3]);
    // $display(dut.residual_inst.header.min_values.bits_required[2:0]);

    // for (int i = 0; i < 32; i++) begin
    //   $display("Pixel %0d: Line: %014b", i, dut.cc_reg.lines.l1.l.pix[i]);
    // end
    $display("==============================================================");
    if (flag !== expected_flag) begin
      $display("ERROR: Output flag does not match expected output.");
      $display("Expected: %b", expected_flag);
      $display("Got:      %b", flag);
    end else begin
      $display("********** Output flag matches expected output. **********");
    end

    if (flag === 2'b00) begin
      $display("Skipping comparison for flag == 2'b00");
      $display("==============================================================");
      return;
    end

    if (lines[511:464] == expected_lines[0][511:464]) begin
      $display("********** Header computation matches. **********");
    end

    if (lines[511:0] == expected_lines[0][511:0]) begin
      $display("********** Output lines match expected output. **********");
    end else begin
     $display("ERROR: Output lines do not match expected output.");
      $display("Expected: %b", expected_lines[0][511:0]);
      $display("Got:      %b", lines[511:0]);
    end
    $display("==============================================================");

  endtask

  initial begin 
    rst = 1'b1;
    # 20;
    rst = 1'b0;
    # 20;

    read_pixels_from_file("/home/kkhulbe2/Documents/cs534/quick-send/hardware/hvl/vcs/pixels.txt");
    read_expected_output("/home/kkhulbe2/Documents/cs534/quick-send/hardware/hvl/vcs/expected_output.txt");
    # 20;
    compare_output();
    # 5;
    $finish;
  end

  initial begin
    clk = 1'b0;
    forever #1 clk = ~clk;
  end 
endmodule // top_tb
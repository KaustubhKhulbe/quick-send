module top_tb;

    timeunit 1ms;
    timeprecision 1ps;

  bit clk;
  bit rst;
  logic [7:0][3:0] pixels [31:0];
  logic              compressable;
  logic [47:0]       h;

  cpu dut(clk, rst, pixels, compressable, h);

    longint timeout;
    initial begin
        $value$plusargs("TIMEOUT_ECE411=%d", timeout);
    end


    initial begin
        $fsdbDumpfile("dump.fsdb");
        $fsdbDumpvars(0, "+all");
        rst = 1'b1;
        repeat (2) @(posedge clk);
        rst <= 1'b0;
    end

    always @(posedge clk) begin
        if (timeout == 0) begin
            $error("TB Error: Timed out");
            $fatal;
        end
        timeout <= timeout - 1;
    end

endmodule

module cpu
(
    input   logic           clk,
    input   logic           rst,

    output  logic           tmp
);

    bit tmp;
    always_ff @(posedge clk) begin
        if (rst) begin
            tmp <= 1'b1;
        end
        else begin
            tmp <= 1'b0;
        end
    end


    
endmodule



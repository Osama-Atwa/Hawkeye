`timescale 1ns / 1ps


module avg_pool_multiple(mat_in_y, clk, rst, avg_out, sum_out, finished);

parameter DATAWIDTH = 32;
parameter MAT_DIMENSION = 2;
parameter CHANNEL_COUNT = 1;
parameter DIVISOR = 32'h3bc1e4bc; //1/169
//parameter DIVISOR = 16'b0_00111_1000001111; //1/169 approx
input wire [DATAWIDTH-1:0] mat_in_y[CHANNEL_COUNT][MAT_DIMENSION][MAT_DIMENSION];
input wire clk, rst;
output wire [DATAWIDTH-1:0] avg_out[CHANNEL_COUNT];
output wire [DATAWIDTH-1:0] sum_out[CHANNEL_COUNT];
output reg finished[CHANNEL_COUNT];

genvar i;
generate
    for(i = 0; i < CHANNEL_COUNT; i++)
    begin
        avg_pool_single #(.DATAWIDTH(DATAWIDTH), .MAT_DIMENSION(MAT_DIMENSION), .DIVISOR(DIVISOR)) pool0(
            .mat_in_y(mat_in_y[i]),
            .clk(clk),
            .rst(rst),
            .avg_out(avg_out[i]),
            .sum_out(sum_out[i]),
            .finished(finished[i])
        );
    end
endgenerate
endmodule

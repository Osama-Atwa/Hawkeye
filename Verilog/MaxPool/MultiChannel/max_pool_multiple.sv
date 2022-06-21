`timescale 1ns / 1ps

module max_pool_multiple(mat_in, clk, rst, mat_out, finished);

parameter DATAWIDTH = 32;
parameter MAT_DIMENSION = 27;

parameter WINDOW_DIMENSION = 3;
parameter STRIDE = 2;
parameter OUTPUT_DIMENSION = (MAT_DIMENSION - WINDOW_DIMENSION)/STRIDE + 1;

parameter CHANNEL_COUNT = 2;


input wire clk, rst;
input wire [DATAWIDTH-1:0] mat_in[CHANNEL_COUNT][MAT_DIMENSION][MAT_DIMENSION];
output reg finished[CHANNEL_COUNT];
output wire [DATAWIDTH-1:0] mat_out[CHANNEL_COUNT][OUTPUT_DIMENSION][OUTPUT_DIMENSION];

genvar i;
generate
    for(i = 0; i < CHANNEL_COUNT; i = i + 1)
    begin
        max_pool_single #(.DATAWIDTH(DATAWIDTH), .MAT_DIMENSION(MAT_DIMENSION), .WINDOW_DIMENSION(WINDOW_DIMENSION), .STRIDE(STRIDE)) pool0(
            .clk(clk),
            .rst(rst),
            .mat_in(mat_in[i]),
            .mat_out(mat_out[i]),
            .finished(finished[i])
        );
    end
endgenerate
endmodule

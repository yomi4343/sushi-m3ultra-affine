// Publication CPU verification harness. Existing upstream tests only.
// Imports are unconditional so --test-filter cannot prune the import seam.
pub const mlx = @import("mlx.zig");
pub const log = @import("log.zig");
pub const io_util = @import("io_util.zig");
comptime {
    _ = @import("transformer.zig");
    _ = @import("qwen4_exp.zig");
    _ = @import("model.zig");
}

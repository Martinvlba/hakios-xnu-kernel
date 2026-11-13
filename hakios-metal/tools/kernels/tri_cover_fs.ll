; tri_cover_fs: tri_cover_vs's fragment function -- the texture sampled at the interpolated coordinates, at the mip
; level the buffer names, through the sampler bound. Hand-written, ours, MIT.
;
;     fragment float4 tri_cover_fs(Out in [[stage_in]], texture2d<float> t [[texture(0)]], sampler s [[sampler(0)]],
;                                  constant Params &p [[buffer(0)]])
;     {
;       float4 c = t.sample(s, in.uv, level(p.level));
;       return p.grey ? c.rrra : c;
;     }
source_filename = "tri_cover_fs"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64-apple-ios16.5.0"

%params = type { [3 x <2 x float>], i32, i32, i8, i8, i8, [13 x i8] }

declare float @air.convert.f.f32.s.i32(i32)
declare { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)

define <{ <4 x float> }> @tri_cover_fs(<4 x float> %position, <2 x float> %uv, ptr addrspace(1) %t, ptr addrspace(2) %s, ptr addrspace(2) %p) local_unnamed_addr {
  %lp = getelementptr inbounds %params, ptr addrspace(2) %p, i64 0, i32 1
  %l = load i32, ptr addrspace(2) %lp, align 8
  %lf = tail call float @air.convert.f.f32.s.i32(i32 %l)
  %r = tail call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) %s, <2 x float> %uv, i1 true, <2 x i32> zeroinitializer, i1 true, float %lf, float 0.000000e+00, i32 0)
  %c = extractvalue { <4 x float>, i8 } %r, 0
  %gp = getelementptr inbounds %params, ptr addrspace(2) %p, i64 0, i32 5
  %g = load i8, ptr addrspace(2) %gp, align 2
  %nog = icmp eq i8 %g, 0
  %grey = shufflevector <4 x float> %c, <4 x float> undef, <4 x i32> <i32 0, i32 0, i32 0, i32 3>
  %out = select i1 %nog, <4 x float> %c, <4 x float> %grey
  %ret = insertvalue <{ <4 x float> }> undef, <4 x float> %out, 0
  ret <{ <4 x float> }> %ret
}

!air.version = !{!0}
!air.language_version = !{!1}
!air.fragment = !{!2}

!0 = !{i32 2, i32 5, i32 0}
!1 = !{!"Metal", i32 3, i32 0, i32 0}
!2 = !{ptr @tri_cover_fs, !3, !5}
!3 = !{!4}
!4 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4", !"air.arg_name", !"color0"}
!5 = !{!6, !7, !8, !9, !10}
!6 = !{i32 0, !"air.position", !"air.center", !"air.no_perspective", !"air.arg_type_name", !"float4", !"air.arg_name", !"position", !"air.arg_unused"}
!7 = !{i32 1, !"air.fragment_input", !"user(locn1)", !"air.center", !"air.perspective", !"air.arg_type_name", !"float2", !"air.arg_name", !"uv"}
!8 = !{i32 2, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"t"}
!9 = !{i32 3, !"air.sampler", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"sampler", !"air.arg_name", !"s"}
!10 = !{i32 4, !"air.buffer", !"air.buffer_size", i32 48, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !11, !"air.arg_type_size", i32 48, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"Params", !"air.arg_name", !"p"}
!11 = !{i32 0, i32 8, i32 3, !"float2", !"uv", i32 24, i32 4, i32 0, !"int", !"level", i32 28, i32 4, i32 0, !"int", !"layer", i32 32, i32 1, i32 0, !"bool", !"flip_x", i32 33, i32 1, i32 0, !"bool", !"flip_y", i32 34, i32 1, i32 0, !"bool", !"grey", i32 35, i32 1, i32 13, !"uchar", !"pad"}

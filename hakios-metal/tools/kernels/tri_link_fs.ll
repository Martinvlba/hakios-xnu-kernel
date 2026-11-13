; tri_link_vs / tri_link_fs: a vertex function that gives SEVEN varyings and a fragment function that takes FOUR of them
; (va, vb, ve, vg -- not vc, vd, vf), as ANGLE's shaders for WebGL do: the two are linked by the names after
; air.vertex_output / air.fragment_input, not by their order. The vertex function covers the target and gives
; ve = (0.25, 0.5, 0.75) and vg = (0.125, 0.375, 0.625), every other varying all ones; the fragment function returns
; (ve.x, vg.y, ve.z, 1) -- 0.25, 0.375, 0.75. Linked by order it would read vc and vd: white.
;
; Hand-written, ours, MIT: nothing in it comes from Apple; only the SHAPE of the metadata is Apple's convention.

source_filename = "tri_link_fs"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64-apple-ios16.5.0"

define <4 x float> @tri_link_fs(<4 x float> %va, <2 x float> %vb, <3 x float> %ve, <3 x float> %vg) local_unnamed_addr {
  %r = extractelement <3 x float> %ve, i32 0
  %g = extractelement <3 x float> %vg, i32 1
  %b = extractelement <3 x float> %ve, i32 2
  %o0 = insertelement <4 x float> <float undef, float undef, float undef, float 1.000000e+00>, float %r, i32 0
  %o1 = insertelement <4 x float> %o0, float %g, i32 1
  %o2 = insertelement <4 x float> %o1, float %b, i32 2
  ret <4 x float> %o2
}

!air.version = !{!0}
!air.language_version = !{!1}
!air.fragment = !{!2}

!0 = !{i32 2, i32 5, i32 0}
!1 = !{!"Metal", i32 3, i32 0, i32 0}
!2 = !{ptr @tri_link_fs, !3, !5}
!3 = !{!4}
!4 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!5 = !{!6, !7, !8, !9}
!6 = !{i32 0, !"air.fragment_input", !"generated(2vaDv4_f)", !"air.center", !"air.perspective", !"air.arg_type_name", !"float4", !"air.arg_name", !"va"}
!7 = !{i32 1, !"air.fragment_input", !"generated(2vbDv2_f)", !"air.center", !"air.perspective", !"air.arg_type_name", !"float2", !"air.arg_name", !"vb"}
!8 = !{i32 2, !"air.fragment_input", !"generated(2veDv3_f)", !"air.center", !"air.perspective", !"air.arg_type_name", !"float3", !"air.arg_name", !"ve"}
!9 = !{i32 3, !"air.fragment_input", !"generated(2vgDv3_f)", !"air.center", !"air.perspective", !"air.arg_type_name", !"float3", !"air.arg_name", !"vg"}

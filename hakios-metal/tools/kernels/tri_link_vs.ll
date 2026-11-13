; tri_link_vs / tri_link_fs: a vertex function that gives SEVEN varyings and a fragment function that takes FOUR of them
; (va, vb, ve, vg -- not vc, vd, vf), as ANGLE's shaders for WebGL do: the two are linked by the names after
; air.vertex_output / air.fragment_input, not by their order. The vertex function covers the target and gives
; ve = (0.25, 0.5, 0.75) and vg = (0.125, 0.375, 0.625), every other varying all ones; the fragment function returns
; (ve.x, vg.y, ve.z, 1) -- 0.25, 0.375, 0.75. Linked by order it would read vc and vd: white.
;
; Hand-written, ours, MIT: nothing in it comes from Apple; only the SHAPE of the metadata is Apple's convention.

source_filename = "tri_link_vs"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64-apple-ios16.5.0"

define <{ <4 x float>, <4 x float>, <2 x float>, <3 x float>, <3 x float>, <3 x float>, <3 x float>, <3 x float> }> @tri_link_vs(i32 %vid) local_unnamed_addr {
  %c4 = icmp eq i32 %vid, 4
  %c3 = icmp eq i32 %vid, 3
  %c2 = icmp eq i32 %vid, 2
  %c1 = icmp eq i32 %vid, 1
  %c0 = icmp eq i32 %vid, 0
  %p4 = select i1 %c4, <4 x float> <float 1.000000e+00, float 1.000000e+00, float 0.000000e+00, float 1.000000e+00>, <4 x float> <float -1.000000e+00, float 1.000000e+00, float 0.000000e+00, float 1.000000e+00>
  %p3 = select i1 %c3, <4 x float> <float 1.000000e+00, float -1.000000e+00, float 0.000000e+00, float 1.000000e+00>, <4 x float> %p4
  %p2 = select i1 %c2, <4 x float> <float -1.000000e+00, float 1.000000e+00, float 0.000000e+00, float 1.000000e+00>, <4 x float> %p3
  %p1 = select i1 %c1, <4 x float> <float 1.000000e+00, float -1.000000e+00, float 0.000000e+00, float 1.000000e+00>, <4 x float> %p2
  %p0 = select i1 %c0, <4 x float> <float -1.000000e+00, float -1.000000e+00, float 0.000000e+00, float 1.000000e+00>, <4 x float> %p1
  %r0 = insertvalue <{ <4 x float>, <4 x float>, <2 x float>, <3 x float>, <3 x float>, <3 x float>, <3 x float>, <3 x float> }> undef, <4 x float> %p0, 0
  %r1 = insertvalue <{ <4 x float>, <4 x float>, <2 x float>, <3 x float>, <3 x float>, <3 x float>, <3 x float>, <3 x float> }> %r0, <4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, 1
  %r2 = insertvalue <{ <4 x float>, <4 x float>, <2 x float>, <3 x float>, <3 x float>, <3 x float>, <3 x float>, <3 x float> }> %r1, <2 x float> <float 1.000000e+00, float 1.000000e+00>, 2
  %r3 = insertvalue <{ <4 x float>, <4 x float>, <2 x float>, <3 x float>, <3 x float>, <3 x float>, <3 x float>, <3 x float> }> %r2, <3 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, 3
  %r4 = insertvalue <{ <4 x float>, <4 x float>, <2 x float>, <3 x float>, <3 x float>, <3 x float>, <3 x float>, <3 x float> }> %r3, <3 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, 4
  %r5 = insertvalue <{ <4 x float>, <4 x float>, <2 x float>, <3 x float>, <3 x float>, <3 x float>, <3 x float>, <3 x float> }> %r4, <3 x float> <float 2.500000e-01, float 5.000000e-01, float 7.500000e-01>, 5
  %r6 = insertvalue <{ <4 x float>, <4 x float>, <2 x float>, <3 x float>, <3 x float>, <3 x float>, <3 x float>, <3 x float> }> %r5, <3 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, 6
  %r7 = insertvalue <{ <4 x float>, <4 x float>, <2 x float>, <3 x float>, <3 x float>, <3 x float>, <3 x float>, <3 x float> }> %r6, <3 x float> <float 1.250000e-01, float 3.750000e-01, float 6.250000e-01>, 7
  ret <{ <4 x float>, <4 x float>, <2 x float>, <3 x float>, <3 x float>, <3 x float>, <3 x float>, <3 x float> }> %r7
}

!air.version = !{!0}
!air.language_version = !{!1}
!air.vertex = !{!2}

!0 = !{i32 2, i32 5, i32 0}
!1 = !{!"Metal", i32 3, i32 0, i32 0}
!2 = !{ptr @tri_link_vs, !3, !12}
!3 = !{!4, !5, !6, !7, !8, !9, !10, !11}
!4 = !{!"air.position", !"air.arg_type_name", !"float4", !"air.arg_name", !"position"}
!5 = !{!"air.vertex_output", !"generated(2vaDv4_f)", !"air.arg_type_name", !"float4", !"air.arg_name", !"va"}
!6 = !{!"air.vertex_output", !"generated(2vbDv2_f)", !"air.arg_type_name", !"float2", !"air.arg_name", !"vb"}
!7 = !{!"air.vertex_output", !"generated(2vcDv3_f)", !"air.arg_type_name", !"float3", !"air.arg_name", !"vc"}
!8 = !{!"air.vertex_output", !"generated(2vdDv3_f)", !"air.arg_type_name", !"float3", !"air.arg_name", !"vd"}
!9 = !{!"air.vertex_output", !"generated(2veDv3_f)", !"air.arg_type_name", !"float3", !"air.arg_name", !"ve"}
!10 = !{!"air.vertex_output", !"generated(2vfDv3_f)", !"air.arg_type_name", !"float3", !"air.arg_name", !"vf"}
!11 = !{!"air.vertex_output", !"generated(2vgDv3_f)", !"air.arg_type_name", !"float3", !"air.arg_name", !"vg"}
!12 = !{!13}
!13 = !{i32 0, !"air.vertex_id", !"air.arg_type_name", !"uint", !"air.arg_name", !"vid"}

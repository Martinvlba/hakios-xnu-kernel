; tri_cover_vs: ONE TRIANGLE OVER THE WHOLE TARGET, texture coordinates and two switches from a constant buffer -- the
; shape of a "blit" vertex function as a GL-on-Metal layer writes it (WebKit's WebGL through ANGLE drew nothing with
; such a pair on the Pixel 8a, 11-13). Hand-written, ours, MIT; only the metadata's shape is Metal's.
;
;     struct Params { float2 uv[3]; int level; int layer; bool flip_x, flip_y, grey; uchar pad[13]; };   // 48 bytes
;     static const float2 corners[3] = { {-1, -1}, {3, -1}, {-1, 3} };
;     struct Out { float4 position [[position]]; float2 uv [[user(locn1)]]; };
;     vertex Out tri_cover_vs(uint vid [[vertex_id]], constant Params &p [[buffer(0)]])
;     {
;       Out o; o.position = float4(corners[vid], 0, 1); o.uv = p.uv[vid];
;       if (p.flip_x) o.position.x = -o.position.x;
;       if (!p.flip_y) o.position.y = -o.position.y;
;       return o;
;     }
;
; What it asks of the front end and the driver: a constant array of float2 indexed by the vertex number, the same of
; a buffer's member, two bool (i8) loads from the buffer, branches joined by phi of a float4, two outputs.
source_filename = "tri_cover_vs"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64-apple-ios16.5.0"

%params = type { [3 x <2 x float>], i32, i32, i8, i8, i8, [13 x i8] }

@corners = internal unnamed_addr addrspace(2) constant [3 x <2 x float>] [<2 x float> <float -1.000000e+00, float -1.000000e+00>, <2 x float> <float 3.000000e+00, float -1.000000e+00>, <2 x float> <float -1.000000e+00, float 3.000000e+00>], align 8

define <{ <4 x float>, <2 x float> }> @tri_cover_vs(i32 %vid, ptr addrspace(2) %p) local_unnamed_addr {
  %i = zext i32 %vid to i64
  %cp = getelementptr inbounds [3 x <2 x float>], ptr addrspace(2) @corners, i64 0, i64 %i
  %c = load <2 x float>, ptr addrspace(2) %cp, align 8
  %c4 = shufflevector <2 x float> %c, <2 x float> undef, <4 x i32> <i32 0, i32 1, i32 undef, i32 undef>
  %pos = shufflevector <4 x float> %c4, <4 x float> <float undef, float undef, float 0.000000e+00, float 1.000000e+00>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>
  %up = getelementptr inbounds %params, ptr addrspace(2) %p, i64 0, i32 0, i64 %i
  %uv = load <2 x float>, ptr addrspace(2) %up, align 8
  %fxp = getelementptr inbounds %params, ptr addrspace(2) %p, i64 0, i32 3
  %fx = load i8, ptr addrspace(2) %fxp, align 8
  %nofx = icmp eq i8 %fx, 0
  br i1 %nofx, label %xdone, label %xflip

xflip:
  %x = extractelement <2 x float> %c, i32 0
  %xn = fsub float -0.000000e+00, %x
  %posx = insertelement <4 x float> %pos, float %xn, i64 0
  br label %xdone

xdone:
  %pos2 = phi <4 x float> [ %posx, %xflip ], [ %pos, %0 ]
  %fyp = getelementptr inbounds %params, ptr addrspace(2) %p, i64 0, i32 4
  %fy = load i8, ptr addrspace(2) %fyp, align 1
  %nofy = icmp eq i8 %fy, 0
  br i1 %nofy, label %yflip, label %ydone

yflip:
  %y = extractelement <4 x float> %pos2, i64 1
  %yn = fsub float -0.000000e+00, %y
  %posy = insertelement <4 x float> %pos2, float %yn, i64 1
  br label %ydone

ydone:
  %pos3 = phi <4 x float> [ %pos2, %xdone ], [ %posy, %yflip ]
  %r0 = insertvalue <{ <4 x float>, <2 x float> }> undef, <4 x float> %pos3, 0
  %r1 = insertvalue <{ <4 x float>, <2 x float> }> %r0, <2 x float> %uv, 1
  ret <{ <4 x float>, <2 x float> }> %r1
}

!air.version = !{!0}
!air.language_version = !{!1}
!air.vertex = !{!2}

!0 = !{i32 2, i32 5, i32 0}
!1 = !{!"Metal", i32 3, i32 0, i32 0}
!2 = !{ptr @tri_cover_vs, !3, !6}
!3 = !{!4, !5}
!4 = !{!"air.position", !"air.arg_type_name", !"float4", !"air.arg_name", !"position"}
!5 = !{!"air.vertex_output", !"user(locn1)", !"air.arg_type_name", !"float2", !"air.arg_name", !"uv"}
!6 = !{!7, !8}
!7 = !{i32 0, !"air.vertex_id", !"air.arg_type_name", !"uint", !"air.arg_name", !"vid"}
!8 = !{i32 1, !"air.buffer", !"air.buffer_size", i32 48, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !9, !"air.arg_type_size", i32 48, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"Params", !"air.arg_name", !"p"}
!9 = !{i32 0, i32 8, i32 3, !"float2", !"uv", i32 24, i32 4, i32 0, !"int", !"level", i32 28, i32 4, i32 0, !"int", !"layer", i32 32, i32 1, i32 0, !"bool", !"flip_x", i32 33, i32 1, i32 0, !"bool", !"flip_y", i32 34, i32 1, i32 0, !"bool", !"grey", i32 35, i32 1, i32 13, !"uchar", !"pad"}

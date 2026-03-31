
GraphDiffusionFlow[R_MeshRegion, steps_, stepsize_, \[Theta]_] := 
 Module[{n, belist, pts, bplist, a, m, aplus, aminus, \[Tau], edges, 
   bedges, solve}, \[Tau] = stepsize;
  n = MeshCellCount[R, 0];
  edges = MeshCells[R, 1, "Multicells" -> True][[1, 1]];
  a = GraphLaplacian[n, edges];
  m = IdentityMatrix[Length[a], SparseArray];
  belist = 
   Random`Private`PositionsOf[
    Length /@ R["ConnectivityMatrix"[1, 2]]["AdjacencyLists"], 1];
  If[Length[belist] > 0, bedges = edges[[belist]];
   bplist = Sort[DeleteDuplicates[Join @@ bedges]];
   a[[bplist]] = GraphLaplacian[n, bedges][[bplist]];
   bedges =.;
   m[[bplist]] = IdentityMatrix[n, SparseArray][[bplist]];
   bplist =.;];
  aplus = m + (\[Tau] \[Theta]) a;
  aminus = m - (\[Tau] (1 - \[Theta])) a;
  pts = MeshCoordinates[R];
  solve = LinearSolve[aplus];
  Do[pts = solve[aminus.pts];, {i, 1, steps}];
  MeshRegion[pts, MeshCells[R, 2, "Multicells" -> True]]]

GraphLaplacian[n_Integer, 
   edges_: List[List[i_Integer, j_Integer] ..]] := 
  With[{A = 
     SparseArray[
      Rule[Join[edges, Transpose[Transpose[edges][[{2, 1}]]]], 
       ConstantArray[1, 2 Length[edges]]], {n, n}]}, 
   SparseArray[DiagonalMatrix[SparseArray[Total[A]]] - A]];


color = Gray;
lighting = 
  Lighting -> {{"Point", {White, Specularity[White]}, {0, 0, 
      3}}}(*{{"Directional",RGBColor[1,.7,.1],{{5,5,4},{5,5,0}}}}*);


plot3DdataClosed[dat_, val_: 0.5, 
   datrange_: {{-1, 1}, {-1, 1}, {-1, 1}}] := 
  Module[{plot, plotsurf, plotdensity, plotall, f, plotregion},
   (*plot=ListContourPlot3D[dat,Mesh\[Rule]Automatic,
   Contours\[Rule]{val},DataRange\[Rule]datrange,
   PlotRange\[Rule]datrange,ContourStyle\[Rule]{Directive[color, 
   Specularity[33]]},BoundaryStyle\[Rule]None, MaxPlotPoints\[Rule]70];
   
   plotsurf=ListSliceContourPlot3D[dat,Cuboid[Sequence@@(Transpose[
   datrange])],Contours\[Rule]{val},DataRange\[Rule]datrange,
   PlotRange\[Rule]datrange,ContourStyle\[Rule]Black, 
   PlotPoints\[Rule]20,PerformanceGoal\[Rule]"Quality",
   BoundaryStyle\[Rule]None,
   ContourShading\[Rule]{Black,None,Directive[color, Specularity[33]],
   Blue}
   ];
   
   
   plotdensity=ListDensityPlot3D[dat,DataRange\[Rule]datrange,
   PlotRange\[Rule]datrange,OpacityFunction\[Rule](If[#<val, 0,1]&),
   ColorFunction\[Rule](If[#<val, Red,Blue]&),
    MaxPlotPoints\[Rule]70, Boxed\[Rule]False, Axes\[Rule]False];
   
   plotall=Show[plot, plotsurf, Lighting\[Rule]lighting, 
   Boxed\[Rule]False, Axes\[Rule]False];
   
   {plotall, Show[DiscretizeGraphics[plotall],
   PlotRange\[Rule]datrange], plotdensity}
   
   *)
   f = ListInterpolation[dat, datrange];
   RegionPlot3D[
    f[x, y, z] > val, {x, datrange[[1, 1]], datrange[[1, 2]]}, {y, 
     datrange[[2, 1]], datrange[[2, 2]]}, {z, datrange[[3, 1]], 
     datrange[[3, 2]]}, Mesh -> Automatic, 
    PlotRange -> datrange,(*PlotStyle\[Rule]{Directive[Red, 
    Specularity[33]]},*)(*BoundaryStyle\[Rule]None, *)
    PlotPoints -> 30, Boxed -> False, Axes -> False]
   ];


importPythonArray[fname_] := Switch[StringTake[fname,-4], ".npy", importNPY[fname], ".pkl", importPKL[fname], _, "ERROR "<>fname];

importPKL[fname_] := ExternalEvaluate["Python", "import pickle; with open('" <> fname <> "', 'rb') as fp: dat=pickle.load(fp); .astype('float32')"];

importNPY[fname_] := ExternalEvaluate["Python", "import numpy as np;np.load('" <> fname <> "').astype('float32')"];

importNPYshell[fname_] :=
  Module[{rawdat, ndim, shape},
   rawdat=Import["! python3 -c 'import numpy as np;a=np.load(\""<>fname<>"\");print(\"\\n\".join(map(str,(a.ndim,)+a.shape)));print(\"\\n\".join(map(str,np.ravel(a).tolist())))'", "Table"][[All,1]];
   ndim = rawdat[[1]];
   shape = rawdat[[2 ;; 1 + ndim]];
   ArrayReshape[rawdat[[2 + ndim ;;]], shape]
   ];



BeginPackage["NumPyArray`"];

Unprotect @@ Names["NumPyArray`*"];
ClearAll @@ Names["NumPyArray`*"];

ReadNumPyArray::usage = 
  "ReadNumPyArray[fileName] loads a NumPy array saved in a .npy file. \
Only numeric data types are supported";
ReadNumPyArray::invalidFile = "`1` is not a valid NPY file";
ReadNumPyArray::invalidVersion = "NPY version `1`.`2` not supported";
ReadNumPyArray::invalidDType = "Invalid dtype `1`";
ReadNumPyArray::fortranOrder = "Fortran order is not supported";

Begin["Private`"];

ReadNumPyArray[file_] := 
  Module[{fs, formatVersion, headerLength, data, pyType, fortranOrder,
     shape, dtype, headerDic, header, byteOrder, retVal}, 
   fs = OpenRead[file, BinaryFormat -> True];
   If[fs === $Failed, fs,(*Check magic string*)
    retVal = 
     If[BinaryReadList[fs, "Character8", 6] == 
       Join[{FromCharacterCode[147]}, Characters["NUMPY"]], 
      formatVersion = BinaryReadList[fs, "Byte", 2];
      (*Check format version*)
      If[formatVersion == {1, 0},(*Read header*)
       headerLength = 
        BinaryRead[fs, "UnsignedInteger16", ByteOrdering -> -1];
       header = 
        StringJoin[BinaryReadList[fs, "Character8", headerLength]];
       headerDic = ParseDic[header];
       dtype = headerDic[["descr"]];
       fortranOrder = headerDic[["fortran_order"]];
       shape = headerDic[["shape"]];
       (*Fortran order is not supported*)
       If[! fortranOrder,(*Endianness*)
        byteOrder = 
         Switch[StringTake[dtype, 1], "<", -1, ">", 
          1, _, $ByteOrdering];
        (*Data type*)pyType = StringTake[dtype, {2, -1}];
        mathematicaDataType = 
         Switch[pyType, "b1", "Integer8", "B1", "UnsignedInteger8", 
          "i1", "Integer8", "u1", "UnsignedInteger8", "i2", 
          "Integer16", "u2", "UnsignedInteger16", "i4", "Integer32", 
          "u4", "UnsignedInteger32", "i8", "Integer64", "u8", 
          "UnsignedInteger64", "i16", "Integer128", "u16", 
          "UnsignedInteger128", "f4", "Real32", "f2", "Real16", "f8", 
          "Real64",(*"f16","Real128",*)"c8", "Complex64", "c16", 
          "Complex128",(*"c32","Complex256",*)_, 
          Message[ReadNumPyArray::invalidDType, pyType]; "Unknown"];
        (*Read data*)
        If[mathematicaDataType == "Unknown", $Failed, 
         data = BinaryReadList[fs, mathematicaDataType, 
           ByteOrdering -> byteOrder];
         ArrayReshape[data, shape]], 
        Message[ReadNumPyArray::fortranOrder]; $Failed], 
       Message[ReadNumPyArray::invalidVersion, formatVersion[[1]], 
        formatVersion[[2]]]; $Failed], 
      Message[ReadNumPyArray::invalidFile, file]; $Failed];
    Close[fs];
    retVal]];

ParseDic[dicString_] := 
  Module[{table}, 
   table = StringCases[dicString, 
     RegularExpression[
       "('.*?(?<!\\\\)')\\s*:\\s*('.*?(?<!\\\\)'|True|False|\\((?:\\d+\
,?\\s*)*\\)|\\d+)"] -> {"$1", "$2"}];
   table = 
    MapAt[If[StringMatchQ[#, "'" ~~ __ ~~ "'"], 
       StringReplace[StringTake[#, {2, -2}], "\\'" -> "'"], #] &, 
     table, {All, All}];
   table = 
    MapAt[If[# == "True" || # == "False", ToExpression[#], #] &, 
     table, {All, All}];
   table = 
    MapAt[If[StringQ[#] && StringMatchQ[#, "(" ~~ __ ~~ ")"], 
       ToExpression[
        StringReplace[
         StringReplace[#, 
          RegularExpression[",\\s*\\)"] -> ")"], {"(" -> "{", 
          ")" -> "}"}]], #] &, table, {All, All}];
   AssociationThread[table[[All, 1]], table[[All, 2]]]];

End[];
EndPackage[];

userArgPos= Flatten[Position[((StringLength[#]>=2)&&(StringTake[#,-2]==".m"))&/@ $CommandLine[[;;-2]], True,1,1]][[1]]+1;
datfiles = Select[$CommandLine[[userArgPos;;]], MemberQ[{".npy", ".pkl"}, StringTake[#,-4]]&];
Print["debug datfiles ", datfiles];
Print["userArgPos ", userArgPos, " cmd ", $CommandLine];
val=ToExpression[$CommandLine[[-1]]];
alldat = (tmp=importPythonArray[#][[;;,;;,;;,;;,;;,1]]; dim=Dimensions[tmp]; ArrayReshape[tmp, Join[{dim[[1]]*dim[[2]]}, dim[[3;;5]]]])&/@datfiles;
ndat= Length[datfiles];
Print["debug alldat", Dimensions[alldat], " ndat ", ndat, " userArgPos ", userArgPos, " datfiles ", datfiles, " CommandLine ", $CommandLine];

(*LaunchKernels[18];*)
ParallelDo[
  g = GraphicsRow[Table[plot3DdataClosed[alldat[[j]][[i]]// Normal, val], {j, ndat}], ImageSize->1200];
  Export["output3d_"<>ToString[NumberForm[i, {5, 0}, NumberPadding -> {"0", "0"}]]<>".png", g];

, {i, Length[alldat[[1]]]}]



CUDA_cuBLAS_DP_implementation
=============================

A conversion of a 64 bit Dynamic Programming problem to a Linear Algebra CUDA implementation.

A serial CPU DP approach and a CUDA cuBLAS approach to the TopCoder problem 'CandyBox';

http://community.topcoder.com/stat?c=problem_statement&pm=10744&rd=14147&rm=&cr=22653720

Wo ist die Liebe Deutschland?  

I see you Chicago! 

This problem has a number of ways it can be solved, and in this case the GPU version uses Linear Algebra (cuBLAS) to raise the probability Matrix to large powers (through exponentiating by squaring method). Since GPUs excel at Linear Algebra, this implementation runs much faster on the GPU than a straight translation of the CPU DP implementation to the CUDA equivalent. A CPU Linear Algebra version would be much slower, so the faster DP CPU version was compared.

A relatively small test was used for now, and there was at least a 86x speedup overl CPU version, but as all variables get larger the relative difference in running time will be larger (favoring the GPU version).


____
<table>
<tr>
    <th>Num Pieces Candy</th><th>Num different Candies</th><th>Num Swaps</th><th>CPU time</th><th>GPU time</th><th>CUDA Speedup</th>
</tr>
  <tr>
    <td>100</td><td>50</td><td>10000</td><td> 86 ms</td><td>1 ms</td><td> 86.x</td>
  </tr>
</table>  
____

NOTE: All CUDA GPU times include all device memsets, host-device memory copies and device-host memory copies.  

Will create larger data sets for bigger tests, and this is just a beta version.  The CUDA implementation can be further optimized as well, but already solves the problem (including all memory ops and copies) in under 1 ms, so that is impressive.
   
War gegen Python!

____

CPU= Intel I-7 3770K 3.5 Ghz with 3.9 Ghz target

GPU= Tesla K20c 5GB

Windows 7 Ultimate x64

Visual Studio 2010 x64  


<script>
  (function(i,s,o,g,r,a,m){i['GoogleAnalyticsObject']=r;i[r]=i[r]||function(){
  (i[r].q=i[r].q||[]).push(arguments)},i[r].l=1*new Date();a=s.createElement(o),
  m=s.getElementsByTagName(o)[0];a.async=1;a.src=g;m.parentNode.insertBefore(a,m)
  })(window,document,'script','//www.google-analytics.com/analytics.js','ga');

  ga('create', 'UA-43459430-1', 'github.com');
  ga('send', 'pageview');

</script>

[![githalytics.com alpha](https://cruel-carlota.pagodabox.com/d40d1ae4136dd45569d36b3e67930e12 "githalytics.com")](http://githalytics.com/OlegKonings/CUDA_vs_CPU_DynamicProgramming_double)
[![githalytics.com alpha](https://cruel-carlota.pagodabox.com/d40d1ae4136dd45569d36b3e67930e12 "githalytics.com")](http://githalytics.com/OlegKonings/CUDA_vs_CPU_DynamicProgramming_double)


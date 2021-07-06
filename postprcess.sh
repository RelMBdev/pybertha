#!/bin/bash


callfunctions () {

  cp ./SHE/plot.p ./SHE/$SYSYEMNAME/
  cp ./SHE/extractintE.py ./SHE/$SYSYEMNAME/
  cp load_cube.py ./SHE/$SYSYEMNAME/
  cd ./SHE/$SYSYEMNAME/


  i=1
  while [ $i -ne 13 ]
  do
	  echo "$i"

	  python3 /home/redo/BERTHA/pycubescd/pycd.py -f pair"$i".cube

	  pymol  -c load_cube.py  pair"$i".cube  $SYSYEMNAME.xyz
	  mv this.png pair"$i".png 

	  pymol  -c load_cube.py  "nocv-"$i".cube"  $SYSYEMNAME.xyz
	  mv this.png "nocv-"$i".png"

	  pymol  -c load_cube.py  "nocv+"$i".cube"  $SYSYEMNAME.xyz
	  mv this.png "nocv+"$i".png"

	  i=$(($i+1))
  done

  pymol  -c load_cube.py diff_tot_ortho.cube $SYSYEMNAME.xyz
  mv this.png diff_tot_ortho.png
  python3 /home/redo/BERTHA/pycubescd/pycd.py -f diff_tot_ortho.cube

  pymol  -c load_cube.py diff_tot.cube $SYSYEMNAME.xyz
  mv this.png diff_tot.png
  python3 /home/redo/BERTHA/pycubescd/pycd.py -f diff_tot.cube 

  gnuplot plot.p
  epstopdf cd.ps 

  cp ../results.tex .

  python3 extractintE.py $SYSYEMNAME"out.txt" >> results.tex

  pdflatex results.tex

  cd -
}

export SYSYEMNAME="AuHg+"
callfunctions

export SYSYEMNAME="AuCn+"
callfunctions

export SYSYEMNAME="AuPb+"
callfunctions

export SYSYEMNAME="AuFl+"
callfunctions

export SYSYEMNAME="AuRn+"
callfunctions

export SYSYEMNAME="AuOg+"
callfunctions


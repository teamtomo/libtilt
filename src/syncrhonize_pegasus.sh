rsync -avlr /home/sanchezg/cryo/myProjects/torchCryoAlign pegasus:/homes/sanchezg/cryo/myProjects/ &&
rsync -avlr /home/sanchezg/sideProjects/tensorCache  pegasus:/homes/sanchezg/sideProjects/ &&
rsync -avlr /home/sanchezg/cryo/sideProjects/starstack  pegasus:~/cryo/sideProjects/  &&
rsync -avlr /home/sanchezg/cryo/tools/libtilt pegasus:~/cryo/tools/


echo "

conda activate ddCryoEM2
cd cryo/myProjects/torchCryoAlign
PYTHONPATH=~/cryo/sideProjects/:~/sideProjects/tensorCache/:~/cryo/tools/libtilt/src/:$PYTHONPATH python -m torchCryoAlign.alignerFourier --reference_vol ~/cryo/data/preAlignedParticles/EMPIAR-10166/data/allparticles_reconstruct.mrc --star_in_fname ~/cryo/data/preAlignedParticles/EMPIAR-10166/data/projections/1000proj.star --star_out_fname /tmp/cmdPruebaAlign.star --particles_root_dir ~/cryo/data/preAlignedParticles/EMPIAR-10166/data --batch_size 24 --grid_resolution_degs 2 --grid_distance_degs 6 --n_cpus_per_job 1 --limit_to_n_particles 4000

"
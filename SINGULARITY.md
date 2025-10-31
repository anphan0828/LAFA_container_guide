# Singularity Container Guide
Singularity is another way to create containers which is more compatible with running on HPCs. Singularity is intended to run on a linux machine, so windows and mac users will need a virtual machine (see [documentation](https://docs.sylabs.io/guides/4.3/admin-guide/installation.html#installation-on-windows-or-mac))

## Singularity install
on linux or linux virtual environment with conda,
```bash
conda install conda-forge::singularity
```

Or you can [download from binaries](https://github.com/sylabs/singularity/blob/v4.3.4/INSTALL.md) or follow the [installation instructions](https://docs.sylabs.io/guides/4.3/admin-guide/installation.html) from the documentation.

## Creating a container
The container can be specified using a `.def` file and the image can be built using: 
```bash
singularity build --fakeroot <container_image>.sif <container_name>.def
```
or
```bash
sudo singularity build <container_image>.sif <container_name>.def
```

### definition file comparison to docker

|                                  | **Docker** | **Singularity**   |
| -------------------------------- | ---------- | ----------------- |
| container definition file        | Dockerfile | \<container>.def  |
| header                           | FROM       | Bootstrap<br>From |
| commands run on host system      |            | %setup            |
| copy files to container          | COPY       | %files            |
| define environment variables     | ENV        | %environment      |
| commands to run while creating   | RUN        | %post             |
| command run when running image   | ENTRYPOINT | %runscript        |
| test at end of build to validate |            | %test             |
| image descriptions               | LABEL      | %labels           |
| image help                       |            | %help             |

There are probably more parallels in commands, I just didn't go into it fully. For a basic comparison, look at the `/nonexp_container` folder. Both the `Dockerfile` and `nonexp.def` files build an equivalently functioning container for docker and singularity respectively.

To make sure that the singularity images behave like the docker images, files should be placed in the `app` directory.

## Running the container
```bash
singularity run -B path/to/data:/hostcwd <container_image>.sif args
```
## nonexp example
navigate to the `nonexp_container` directory
```bash
# build
singularity build --fakeroot nonexp_image.sif nonexp.def

# run
singularity run -B ../test_data:/app/data nonexp_image.sif \
  --annot_file /app/data/goa_uniprot_selected.gaf \
  --selected_go Computational,Phylogenetical,Authorstatement,Curatorstatement,Electronic \
  --query_file /app/data/test_sequences.fasta \
  --graph /app/data/go-basic-20250601.obo \
  --output_baseline /app/data/train_terms.tsv
```
## Docker --> Singularity container
### Docker images on dockerhub
```bash
sudo singularity build my_container.sif docker://<repo>
```
### local docker images
```bash
docker save <registry_url>/<repo>:<tag> -o my_image.tar
singularity build my_container.sif docker-archive://my_image.tar
```

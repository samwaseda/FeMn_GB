import numpy as np
from collections import defaultdict
from pyiron_atomistics import Project as PyironProject
import pint
from tqdm.auto import tqdm


class Project(PyironProject):
    def __init__(self, *args, **vargs):
        super().__init__(*args, **vargs)
        self.lammps = LammpsGB(self)
        self.bulk = Bulk(self)
        self.grain_boundary = GrainBoundary(self)


class LammpsGB:
    def __init__(self, project, max_sigma=100):
        self.project = project
        self.potential = '1997--Ackland-G-J--Fe--LAMMPS--ipr1'
        self.max_sigma = 100
        self.max_axis = 4
        self.n_max = 100
        self.cutoff_radius = 5
        self._unique_gb = None
        self._list_gb = None
        self.n_mesh_points = 11

    @property
    def bulk(self):
        a_0 = self._job_bulk.get_structure().cell[0, 0]
        return self.project.create.structure.bulk('Fe', cubic=True, a=a_0)

    @property
    def _job_bulk(self):
        return self.get_job('bulk', self.project.create.structure.bulk('Fe', cubic=True))

    def get_job(self, job_name, structure, minimize=True, run=True):
        lmp = self.project.create.job.Lammps(job_name)
        lmp.structure = structure
        lmp.potential = self.potential
        if minimize:
            lmp.calc_minimize(pressure=[0, 0, 0])
        if lmp.status.initialized and run:
            lmp.run()
        return lmp

    @property
    def mu(self):
        return self._job_bulk['output/generic/energy_pot'][-1] / 2

    def get_lmp_gb(self, axis, sigma, plane, target_width=30):
        return self._get_lmp_gb(
            axis=axis,
            sigma=sigma,
            plane=plane,
            target_width=target_width,
        )[0]

    @property
    def _meshgrid(self):
        mesh = np.meshgrid(
            *2 * [np.arange(self.n_mesh_points) / self.n_mesh_points]
        )
        return np.stack(mesh, axis=-1).reshape(-1, 2)

    def _get_gamma_min(self, lmp, mesh):
        structure = lmp.structure.copy()
        cell_init = lmp.structure.cell.copy()
        if lmp.status.initialized:
            with lmp.interactive_open() as job:
                layers = structure.analyse.get_layers(planes=[1, 0, 0])
                for xx in mesh * structure.cell.diagonal()[1:]:
                    job.structure = structure.copy()
                    job.structure.positions[layers < layers.max() / 2, 1:] += xx
                    job.run()
        E, indices = np.unique(
            np.round(lmp.output.energy_pot, decimals=8), return_index=True
        )
        structures = [lmp.get_structure(i) for i in indices]
        for structure in structures:
            structure.set_cell(cell_init, scale_atoms=True)
        return E, structures

    def _get_lmp_gb(self, axis, sigma, plane, target_width=30):
        gb = self.project.create.structure.aimsgb.build(
            axis=axis, sigma=sigma, plane=plane, initial_struct=self.bulk
        )
        repeat = np.max([np.rint(target_width / gb.cell.diagonal().max()), 1]).astype(int)
        E_min = np.inf
        if repeat * len(gb) > self.n_max:
            return
        for i in range(2):
            for j in range(2):
                structure = self.project.create.structure.aimsgb.build(
                    axis=axis,
                    sigma=sigma,
                    plane=plane,
                    uc_a=repeat,
                    uc_b=repeat,
                    delete_layer=f'{i}b{j}t{i}b{j}t',
                    initial_struct=self.bulk
                )
                lmp = self.get_job(
                    ('lmp_gb', *axis, sigma, *plane, i, j), structure, run=False
                )
                E, struct = self._get_gamma_min(lmp, self._meshgrid)
                E_current = E - len(structure) * self.mu
                if i + j == 0 or np.min(E_min) > np.min(E_current) + 1.0e-3:
                    E_min = E_current
                    min_structure = struct
        return min_structure, E_min

    @property
    def list_gb(self):
        if 'structure' not in list(self.unique_gb.keys()):
            for sigma, plane, axis in zip(
                self.unique_gb['sigma'],
                self.unique_gb['plane'],
                self.unique_gb['axis']
            ):
                str_e = self._get_lmp_gb(axis, sigma, plane)
                if str_e is None:
                    continue
                self._unique_gb['structure'].append(str_e[0])
                self._unique_gb['energy'].append(str_e[1])
        return self.unique_gb

    def _get_unique_gb(
        self, initial_struct, max_axis=4, max_sigma=100, cutoff_radius=5
    ):
        list_gb = defaultdict(list)
        axes = np.stack(
            np.meshgrid(*3 * [np.arange(max_axis)]), axis=-1
        ).reshape(-1, 3)
        axes = axes[np.gcd.reduce(axes, axis=-1) == 1]
        for axis in tqdm(axes):
            for k, v in self.project.create.structure.aimsgb.info(
                axis, max_sigma
            ).items():
                for p in np.asarray(v['plane'])[:, 0]:
                    structure = self.project.create.structure.aimsgb.build(
                        axis=axis, sigma=k, plane=p, initial_struct=initial_struct
                    )
                    if structure is None or len(structure) > 100:
                        continue
                    E = np.sum(
                        1 / structure.get_neighbors(
                            num_neighbors=None, cutoff_radius=cutoff_radius
                        ).flattened.distances
                    )
                    list_gb['energy'].append(E)
                    list_gb['axis'].append(axis)
                    list_gb['plane'].append(p)
                    list_gb['sigma'].append(k)
        _, indices = np.unique(
            np.round(list_gb['energy'], decimals=8), return_index=True
        )
        del list_gb['energy']
        for k, v in list_gb.items():
            list_gb[k] = np.asarray(v)[indices]
        return list_gb

    @property
    def unique_gb(self):
        if self._unique_gb is None:
            self._unique_gb = self._get_unique_gb(
                initial_struct=self.bulk,
                max_axis=self.max_axis,
                max_sigma=self.max_sigma,
                cutoff_radius=self.cutoff_radius
            )
        return self._unique_gb


def set_parameters(spx, n_cores=40, queue='cm', random=True):
    spx.set_convergence_precision(electronic_energy=1e-6)
    spx.set_kpoints(k_mesh_spacing=0.1)
    spx.set_encut(500)
    magmoms = 2 * np.ones(len(spx.structure))
    ind_Mn = spx.structure.select_index('Mn')
    if len(ind_Mn) == 1:
        magmoms[ind_Mn] *= -1
    if len(spx.structure.select_index('Mn')) > 1 and random:
        magmoms[ind_Mn] *= np.random.choice([1, -1], len(ind_Mn))
    spx.structure.set_initial_magnetic_moments(magmoms)
    spx.set_occupancy_smearing(smearing='FermiDirac')
    spx.set_mixing_parameters(density_residual_scaling=0.1, spin_residual_scaling=0.1)
    spx.server.queue = queue
    spx.server.cores = n_cores
    spx.calc_minimize()
    return spx


def get_values(neigh, species):
    denom = np.unique(neigh.flattened.shells, return_counts=True)[1]
    count = np.zeros(len(denom))
    prod = species[neigh.flattened.atom_numbers] * 2 - 1
    prod *= species[neigh.flattened.indices] * 2 - 1
    np.add.at(count, neigh.flattened.shells - 1, prod)
    return np.linalg.norm(count / denom - (1 - 2 * np.mean(species))**2)**2


def get_sqs(structure, steps=1000, num_neighbors=58):
    species = structure.get_chemical_indices().copy()
    neigh = structure.get_neighbors(num_neighbors=num_neighbors)
    current_value = get_values(neigh, species)
    for i in range(steps):
        ind_Fe = np.random.choice(np.where(species == 0)[0])
        ind_Mn = np.random.choice(np.where(species == 1)[0])
        species[ind_Fe], species[ind_Mn] = species[ind_Mn], species[ind_Fe]
        new_value = get_values(neigh, species)
        if new_value > current_value:
            species[ind_Fe], species[ind_Mn] = species[ind_Mn], species[ind_Fe]
        else:
            current_value = new_value
    structure[:] = 'Fe'
    structure[np.where(species == 1)[0]] = 'Mn'
    return structure


class Bulk:
    def __init__(self, project):
        self.project = project

    def run_murnaghan(self):
        spx = self.project.create.job.Sphinx('bulk_Fe')
        spx.structure = self.project.create.structure.crystal('Fe', 'bcc', 2.83)
        spx = set_parameters(spx, n_cores=4)
        murn = spx.create_job('Murnaghan', 'murn_Fe')
        if murn.status.initialized:
            murn.run()

    @property
    def lattice_constant(self):
        murn = self.project.load('murn_Fe')
        if murn is None:
            self.run_murnaghan()
            raise ValueError('Wait for lattice constant to be ready')
        return murn['output/equilibrium_volume']**(1 / 3)

    def get_energy(self, element, n_repeat=3, n_Mn=1):
        if element == 'Fe':
            murn = self.project.load('murn_Fe')
            if murn is None:
                self.run_murnaghan()
                return None
            return murn['output/equilibrium_energy'] / 2
        elif element == 'Mn':
            murn = self.project.load('murn_sqs_{}_{}'.format(n_repeat, n_Mn))
            if murn is None:
                self.run_murnaghan()
                return None
            indices = murn['output/structure/indices']
            N_Fe = np.sum(np.array(murn['output/structure/species'])[indices] == 'Fe')
            coeff = np.polyfit(murn['output/volume'], murn['output/energy'], 3)
            return np.polyval(
                coeff, self.lattice_constant**3 * len(indices) / 2
            ) - N_Fe * self.get_energy('Fe')
        else:
            raise ValueError(element, 'not recognized')

    def run_sqs(self, max_Mn_fraction=0.2):
        for n_repeat in [2, 3]:
            n_atoms = n_repeat**3 * 2
            for n_Mn in np.arange(1, int(n_atoms * max_Mn_fraction)):
                spx = self.project.create.job.Sphinx(('spx_sqs', n_repeat, n_Mn))
                structure = self.project.create.structure.crystal(
                    'Fe', 'bcc', self.lattice_constant
                ).repeat(n_repeat)
                structure[np.random.choice(n_atoms, n_Mn, replace=False)] = 'Mn'
                if np.min([n_atoms - n_Mn, n_Mn]) > 3:
                    structure = get_sqs(structure)
                spx.structure = structure
                spx = set_parameters(spx, n_cores=40 + 40 * (n_repeat - 2))
                spx.run()


class GrainBoundary:
    def __init__(self, project):
        self.project = project
        self._energy_dict = None
        self._structure_dict = {}
        self.symprec = 1.0e-2
        self._segregation_energy = None
        self.temperature = 600
        self.concentration = 10
        self.unit = pint.UnitRegistry()

    def job_table(self, full_table=True):
        return self.project.job_table(full_table=full_table, job='spx_gb_*')

    @property
    def job_names(self):
        jobs = self.job_table().job
        jobs = [job_name for job_name in jobs if 'restart' not in job_name]
        return np.unique(['_'.join(j.split('_')[:-1]) for j in jobs])

    @staticmethod
    def _get_next_job_name(job_name):
        if 'restart' not in job_name:
            return job_name + '_restart_0'
        job_name = job_name.split('_')
        job_name[-1] = str(int(job_name[-1]) + 1)
        return '_'.join(job_name)

    @property
    def energy_dict(self):
        if self._energy_dict is None:
            self._energy_dict = defaultdict(list)
            E_Fe = self.project.bulk.get_energy('Fe')
            for job_type in self.job_names:
                if any([
                    s in list(self.project.job_table(job=f'{job_type}*').status)
                    for s in ['submitted', 'running']
                ]):
                    print(job_type, 'running')
                    continue
                if any([k.startswith(job_type) for k in self._energy_dict.keys()]):
                    continue
                all_jobs = list(self.project.job_table().job)
                for spx in self.project.iter_jobs(
                    job=f'{job_type}*', progress=False
                ):
                    if self._get_next_job_name(spx.job_name) in all_jobs:
                        continue
                    print(spx.job_name)
                    LL = np.diagonal(spx['output/generic/cells'][-1])
                    EE = E_Fe * len(spx['input/structure'])
                    self._energy_dict[job_type].append([
                        np.max(LL),
                        (spx['output/generic/energy_pot'][-1] - EE) / np.prod(np.sort(LL)[:2]) / 2
                    ])
            for k, v in self._energy_dict.items():
                self._energy_dict[k] = np.array(v)[np.argsort(v, axis=0)[:, 0]]
        return self._energy_dict

    @staticmethod
    def _get_fit(x, y, order=3):
        coeff = np.polyfit(x, y, 3)
        c = np.polyder(coeff)
        return -0.5 * c[1] / c[0] - np.sqrt((c[1] / 2 / c[0])**2 - c[2] / c[0]), coeff

    @property
    def structures(self):
        if len(self._structure_dict) == 0:
            for k, v in self.energy_dict.items():
                L, coeff = self._get_fit(*v.T, order=3)
                structure = self.project.load('{}_0d0'.format(k)).structure
                cell = structure.cell.flatten()
                cell[cell.argmax()] = L
                structure.set_cell(cell.reshape(3, 3), scale_atoms=True)
                self._structure_dict[k] = structure.copy()
        return self._structure_dict

    def get_gb_energy(self, J_per_m2=False):
        results = {}
        for k, v in tqdm(self.energy_dict.items()):
            L, coeff = self._get_fit(*v.T, order=3)
            E = np.polyval(coeff, L) * self.unit.electron_volt / self.unit.angstrom**2
            if J_per_m2:
                E = E.to(self.unit.joule / self.unit.meter**2)
            results[k] = E.magnitude
        return results

    def get_angles(self):
        results = {}
        for job_name in self.energy_dict.keys():
            axis = [int(n) for n in job_name.split('_')[2].split('d')]
            sigma = int(job_name.split('_')[3])
            plane = []
            ss = ''
            for s in job_name.split('_')[4].replace('m', '-'):
                ss += s
                try:
                    plane.append(int(ss))
                    ss = ''
                except ValueError:
                    pass
            entry = self.project.create.structure.aimsgb.info(axis, sigma)[sigma]
            results[job_name] = np.array(entry['theta'])[
                np.all(np.array(entry['plane']) == plane, axis=-1).any(axis=-1)
            ].min()
        return results

    def _get_job_energy(self, job):
        conv = np.array([len(e) < 100 for e in job['output/generic/dft/scf_energy_free']])
        forces = np.linalg.norm(job['output/generic/forces'], axis=-1).max(axis=-1)
        try:
            if len(conv) - len(forces) == 1:
                conv[-1] = False
            if len(conv) != len(forces) and forces[np.where(conv)[0][-1]] > 0.01:
                print('max force of', job.job_name, ':', forces[np.where(conv)[0][-1]])
        except IndexError:
            raise IndexError('Index Error on ' + job.job_name)
        return job['output/generic/energy_pot'][conv][-1]

    def _get_segregation_energy(self, job_name, structure):
        E_lst = np.zeros(len(structure))
        equivalent_atoms = structure.get_symmetry(symprec=self.symprec).arg_equivalent_atoms
        gb_energy = self.get_gb_energy()[job_name] * np.sort(structure.cell.diagonal())[:2].prod()
        E_Fe = self.project.bulk.get_energy('Fe') * (len(structure) - 1)
        E_ref = 2 * gb_energy + E_Fe + self.project.bulk.get_energy('Mn', n_repeat=3)
        all_jobs = list(self.project.job_table().job)
        for atom_id in np.unique(equivalent_atoms):
            job_name_Mn = '{}_{}'.format(job_name.replace('gb', 'gbMn'), atom_id)
            if len(self.project.job_table(job=job_name_Mn)) == 0:
                spx = self.project.create.job.Sphinx((job_name.replace('gb', 'gbMn'), atom_id))
                spx.structure = structure.copy()
                spx.structure[atom_id] = 'Mn'
                set_parameters(spx)
                spx.run()
                continue
            while self._get_next_job_name(job_name_Mn) in all_jobs:
                job_name_Mn = self._get_next_job_name(job_name_Mn)
            if np.any([
                s in ['running', 'submitted']
                for s in self.project.job_table(job=job_name_Mn).status
            ]):
                print(job_name_Mn, 'running')
                continue
            spx = self.project.load(job_name_Mn)
            E = self._get_job_energy(spx) - E_ref
            E_lst[equivalent_atoms == atom_id] = E
        return E_lst

    @property
    def segregation_energy(self):
        if self._segregation_energy is None:
            self._segregation_energy = {}
            for job_name, structure in self.structures.items():
                E_lst = self._get_segregation_energy(job_name, structure)
                if np.all(E_lst != 0):
                    self._segregation_energy[job_name] = E_lst
        return self._segregation_energy

    @property
    def c_0(self):
        return self.concentration / 100

    @property
    def _celsius(self):
        return self.temperature + 273

    @property
    def kBT(self):
        return (
            self._celsius * self.unit.kelvin * self.unit.boltzmann_constant
        ).to('eV').magnitude

    def get_occ_probability(self, E):
        return 1 / (1 + (1 - self.c_0) / self.c_0 * np.exp(E / self.kBT))

    def run_collective(self):
        for job_name, E in self.segregation_energy.items():
            for i in range(10):
                structure = self.structures[job_name].copy()
                structure[np.random.random(len(E)) < self.get_occ_probability(E)] = 'Mn'
                for j in range(7):
                    spx = self.project.create.job.Sphinx((
                        job_name.replace('gb', 'gbMnMn'),
                        self.temperature, self.concentration, i, j
                    ))
                    if not spx.status.initialized:
                        continue
                    print(spx.job_name)
                    spx.structure = structure.copy()
                    set_parameters(spx, n_cores=40)
                    spx.run()

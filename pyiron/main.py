import numpy as np
from collections import defaultdict
from pyiron_atomistics import Project as PyironProject


class Project(PyironProject):
    def __init__(
        self,
        path='',
        user=None,
        sql_query=None,
        default_working_directory=False,
    ):
        super().__init__(
            path=path,
            user=user,
            sql_query=sql_query,
            default_working_directory=default_working_directory,
        )
        self.lammps = LammpsGB(self)
        self.bulk = Bulk(self)
        self.grain_boundary = GrainBoundary(self)


class LammpsGB:
    def __init__(self, project, max_sigma=100):
        self.project = project
        self.potential = '1997--Ackland-G-J--Fe--LAMMPS--ipr1'
        self.max_sigma = 100
        self.n_max = 100
        self._list_gb = None

    @property
    def bulk(self):
        a_0 = self._job_bulk.get_structure().cell[0, 0]
        return self.project.create.structure.bulk('Fe', cubic=True, a=a_0)

    @property
    def _job_bulk(self):
        return self.get_job('bulk', self.project.create.structure.bulk('Fe', cubic=True))

    def get_job(self, job_name, structure):
        lmp = self.project.create.job.Lammps(job_name)
        lmp.structure = structure
        lmp.potential = self.potential
        lmp.calc_minimize(pressure=0)
        if lmp.status.initialized:
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
                lmp = self.project.create.job.Lammps(('lmp_gb', *axis, sigma, *plane, i, j))
                if lmp.status.initialized:
                    lmp.potential = self.potential
                    lmp.structure = structure
                    lmp.calc_minimize(pressure=0)
                    lmp.run()
                E_current = lmp.output.energy_pot[-1] - len(structure) * self.mu
                if i + j == 0 or E_min > E_current + 1.0e-3:
                    E_min = E_current
                    min_structure = lmp.get_structure()
                    min_structure.set_cell(lmp.output.cells[0], scale_atoms=True)
        return min_structure, E_min

    @property
    def list_gb(self):
        if self._list_gb is None:
            self._list_gb = defaultdict(list)
            for ix in range(4):
                for iy in range(ix + 1):
                    for iz in range(iy + 1):
                        axis = [ix, iy, iz]
                        if np.gcd.reduce(axis) != 1:
                            continue
                        for k, v in self.project.create.structure.aimsgb.info(
                            axis, self.max_sigma
                        ).items():
                            for p in np.unique(np.reshape(v['plane'], (-1, 3)), axis=0):
                                str_e = self._get_lmp_gb(axis, k, p)
                                if str_e is None:
                                    continue
                                self._list_gb['energy'].append(str_e[1])
                                self._list_gb['axis'].append(axis)
                                self._list_gb['plane'].append(p)
                                self._list_gb['sigma'].append(k)
                                self._list_gb['structure'].append(str_e[0])
            self._list_gb['id'] = np.unique(
                np.round(self._list_gb['energy'], decimals=5), return_inverse=True
            )[1]
        return self._list_gb


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
        murn = self.project.inspect('murn_Fe')
        if murn is None:
            self.run_murnaghan()
            raise ValueError('Wait for lattice constant to be ready')
        return murn['output/equilibrium_volume']**(1 / 3)

    def get_energy(self, element, n_repeat=3, n_Mn=1):
        if element == 'Fe':
            murn = self.project.inspect('murn_Fe')
            if murn is None:
                self.run_murnaghan()
                return None
            return murn['output/equilibrium_energy'] / 2
        elif element == 'Mn':
            murn = self.project.inspect('murn_sqs_{}_{}'.format(n_repeat, n_Mn))
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
                spx.calc_minimize()
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

    def job_table(self, full_table=True):
        jt = self.project.job_table(full_table=full_table)
        return jt[jt.job.str.startswith('spx_gb_')]

    @property
    def job_names(self):
        jobs = self.job_table().job
        jobs = [job_name for job_name in jobs if not job_name.endswith('_restart')]
        return np.unique(['_'.join(j.split('_')[:-1]) for j in jobs])

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
                    continue
                if any([k.startswith(job_type) for k in self._energy_dict.keys()]):
                    continue
                for spx in self.project.iter_jobs(
                    job=f'{job_type}*', convert_to_object=False, progress=False
                ):
                    if len(self.project.job_table(job=f'{spx.job_name}_restart')) > 0:
                        continue
                    LL = np.diagonal(spx['output/generic/cells'][-1])
                    EE = E_Fe * len(spx['input/structure/indices'])
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
        for k, v in self.energy_dict.items():
            L, coeff = self._get_fit(*v.T, order=3)
            E = np.polyval(coeff, L)
            if J_per_m2:
                E *= 16.0219
            results[k] = E
        return results

    def get_angles(self):
        results = {}
        for job_name in self.energy_dict.keys():
            axis = [int(n) for n in job_name.split('_')[2].split('c')]
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

    def _get_segregation_energy(self, job_name, structure):
        E_lst = np.zeros(len(structure))
        equivalent_atoms = structure.get_symmetry(symprec=self.symprec).arg_equivalent_atoms
        gb_energy = self.get_gb_energy()[job_name] * structure.cell.diagonal().prod() / structure.cell.max()
        E_Fe = self.project.bulk.get_energy('Fe') * (len(structure) - 1)
        E_ref = 2 * gb_energy + E_Fe + self.project.bulk.get_energy('Mn', n_repeat=3)
        for atom_id in np.unique(equivalent_atoms):
            job_name_Mn = '{}_{}'.format(job_name.replace('gb', 'gbMn'), atom_id)
            if len(self.project.job_table(job=job_name_Mn)) == 0:
                spx = self.project.create.job.Sphinx((job_name.replace('gb', 'gbMn'), atom_id))
                spx.structure = structure.copy()
                spx.structure[atom_id] = 'Mn'
                set_parameters(spx)
                spx.run()
                continue
            if len(self.project.job_table(job=f'{job_name_Mn}_restart')) > 0:
                job_name_Mn = job_name_Mn + '_restart'
            if np.any([
                s in ['running', 'submitted']
                for s in self.project.job_table(job=job_name_Mn).status
            ]):
                continue
            spx = self.project.inspect(job_name_Mn)
            E = spx['output/generic/energy_pot'][-1] - E_ref
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

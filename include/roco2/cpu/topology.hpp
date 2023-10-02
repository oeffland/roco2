#ifndef INCLUDE_ROCO2_CPU_TOPOLOGY_HPP
#define INCLUDE_ROCO2_CPU_TOPOLOGY_HPP

#include <mpi.h>
#include <roco2/exception.hpp>
#include <roco2/log.hpp>

#include <omp.h>

#include <cstdint>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace roco2
{

namespace cpu
{

    namespace detail
    {

        inline auto parse_list(std::string list) -> std::set<uint32_t>
        {
            std::stringstream s;
            s << list;

            std::set<uint32_t> res;

            std::string part;
            while (std::getline(s, part, ','))
            {
                auto pos = part.find('-');
                if (pos != std::string::npos)
                {
                    // is a range
                    uint32_t from = std::stoi(part.substr(0, pos));
                    uint32_t to = std::stoi(part.substr(pos + 1));

                    for (auto i = from; i <= to; ++i)
                        res.insert(i);
                }
                else
                {
                    // single value
                    res.insert(std::stoi(part));
                }
            }

            return res;
        }
    } // namespace detail

    class topology
    {

        void read_cores()
        {
            std::string online_list;
            std::string present_list;

            {
                std::ifstream cpu_online(base_path + "/online");
                std::getline(cpu_online, online_list);

                std::ifstream cpu_present(base_path + "/present");
                std::getline(cpu_present, present_list);
            }

            auto online = detail::parse_list(online_list);
            auto present = detail::parse_list(present_list);

            for (auto coreid : present)
            {
                if (online.count(coreid) == 1)
                {
                    cores_.push_back(core::read_from_sys(coreid));
                }
                else
                {
                    cores_.emplace_back(coreid);
                }
            }
        }

        void read_sockets()
        {
            for (auto& core : cores_)
            {
                bool found = false;

                for (auto& socket : sockets_)
                {
                    if (socket.id == core.socket)
                    {
                        found = true;
                        socket.cores.push_back(core.id);
                    }
                }

                if (!found)
                {
                    socket s;
                    s.id = core.socket;
                    s.cores.push_back(core.id);

                    sockets_.push_back(s);
                }
            }
        }

        void read_nodes()
        {
            // You're not running on a homogeneous system? Well, may god be
            // kind on your soul, because I won't. 
            
            int size;
            MPI_Comm_size(MPI_COMM_WORLD, &size);


            std::vector<uint32_t> sockets;

            for(auto& socket : sockets_) {
                sockets.push_back(socket.id);
            }

            node n;
            n.sockets = sockets;

            for (int id = 0; id < size; id++)
            {
                n.id = id;
                // yes, I'm not ashamed of myself.
                nodes_.push_back(n);
            }
        }

    public:
        struct core
        {
            uint32_t id;
            uint32_t socket;
            bool online;

            core(uint32_t id = 0, uint32_t socket = static_cast<uint32_t>(-1), bool online = false)
            : id(id), socket(socket), online(online)
            {
            }

            static core read_from_sys(uint32_t id)
            {
                std::stringstream filename_stream;

                filename_stream << base_path << "/cpu" << id << "/topology";

                std::string topology = filename_stream.str();

                std::ifstream package(topology + "/physical_package_id");

                core result;

                package >> result.socket;

                result.online = true;
                result.id = id;

                if (!package)
                {
                    raise("Couldn't read topology of core ", id);
                }

                return result;
            }
        };

        struct socket
        {
            uint32_t id;
            std::vector<uint32_t> cores;
        };

        struct node
        {
            uint32_t id;
            std::vector<uint32_t> sockets;
        };

    private:
        topology()
        {
            read_cores();
            read_sockets();
            read_nodes();

            if (static_cast<int>(num_cores()) != omp_get_max_threads())
            {
                log::warn() << "There's something fishy going on here...";
                log::warn() << "Number of cores (" << num_cores()
                            << ") differ from the number of OMP threads (" << omp_get_max_threads()
                            << ").";
            }
        }

    public:
        static topology& instance()
        {
            static topology t;

            return t;
        }

        std::vector<uint32_t> on_socket(std::uint32_t socket) const
        {
            return get_socket(socket).cores;
        }

        std::uint32_t socket_of(std::uint32_t core) const
        {
            return get_core(core).socket;
        }

        const socket& get_socket(std::uint32_t id) const
        {
            for (auto& s : sockets_)
            {
                if (s.id == id)
                    return s;
            }

            raise("Given socket ", id, " doesn't exists.");
        }

        const core& get_core(std::uint32_t id) const
        {
            for (auto& c : cores_)
            {
                if (c.id == id)
                    return c;
            }

            raise("Given core ", id, " doesn't exists.");
        }

        const node& get_node(std::uint32_t id) const
        {
            for (auto& n : nodes_)
            {
                if (n.id == id)
                    return n;
            }

            raise("Given core ", id, " doesn't exists.");
        }

        std::size_t num_nodes() const
        {
            return nodes_.size();
        }

        std::size_t num_cores() const
        {
            return cores_.size();
        }

        std::size_t num_sockets() const
        {
            return sockets_.size();
        }

        std::size_t num_per_socket(std::uint32_t socket = 0) const
        {
            return get_socket(socket).cores.size();
        }

        const std::vector<core>& cores() const
        {
            return cores_;
        }

        const std::vector<socket>& sockets() const
        {
            return sockets_;
        }

        const std::vector<node>& nodes() const
        {
            return nodes_;
        }

        friend struct core;

    private:
        std::vector<core> cores_;
        std::vector<socket> sockets_;
        std::vector<node> nodes_;

        const static std::string base_path;
    };
} // namespace cpu
} // namespace roco2

#endif // INCLUDE_ROCO2_CPU_TOPOLOGY_HPP

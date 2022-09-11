defmodule Sar do
  import Nx.Defn

  defn range_compression(input) do
    input
    |> Nx.fft()
    |> then(& {&1, Nx.real(&1), Nx.imag(&1)})
    |> then(fn {c, r, i} -> {c, Nx.complex(r, Nx.negate(i))} end)
    |> then(fn {a, b} -> Nx.multiply(a, b) end)
    |> then(& Nx.ifft(&1))
  end
end

inputs =
  [1024, 4096, 32768]
  |> Enum.map(fn size ->
    {
      "size #{size}: IPS #{div(105_000_000, size) |> Number.Delimit.number_to_delimited()}",
      Nx.random_uniform({size}, 0, 65536, type: {:s, 32})
    }
  end)
  |> Map.new()

fft_exla_cpu = EXLA.jit(&Nx.fft/1)
ifft_exla_cpu = EXLA.jit(&Nx.ifft/1)
rc_exla_cpu = EXLA.jit(&Sar.range_compression/1)

Benchee.run(
  %{
    "FFT (Nx)" => fn input -> Nx.fft(input) end,
    "FFT (EXLA CPU)" => fn input -> fft_exla_cpu.(input) end,
    "IFFT (Nx)" => {
      fn input -> Nx.ifft(input) end,
      before_each: fn input -> fft_exla_cpu.(input) end
    },
    "IFFT (EXLA CPU)" => {
      fn input -> ifft_exla_cpu.(input) end,
      before_each: fn input -> fft_exla_cpu.(input) end
    },
    "Range compression (Nx)" => fn input -> Sar.range_compression(input) end,
    "Range compression (EXLA CPU)" => fn input -> rc_exla_cpu.(input) end
  },
  inputs: inputs
)

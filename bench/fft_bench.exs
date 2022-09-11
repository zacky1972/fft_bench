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

Benchee.run(
  %{
    "FFT (Nx)" => fn input -> Nx.fft(input) end,
    "FFT (EXLA CPU)" => fn input -> fft_exla_cpu.(input) end
  },
  inputs: inputs
)

package config

import com.typesafe.config.{Config, ConfigFactory}

import scala.util.Properties

class ServiceConfig(fileNameOption: Option[String] = None) {

  val config: Config = fileNameOption.fold(ifEmpty = ConfigFactory.load())(
    file => ConfigFactory.load(file)
  )

  def envOrElseConfig(name: String): String = {
    config.resolve()
    Properties.envOrElse(
      name.toUpperCase.replaceAll("""\.""", "_"),
      config.getString(name)
    )
  }
}
